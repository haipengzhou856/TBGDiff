from transformers import SegformerForSemanticSegmentation
import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .loss_func import lovasz_hinge
from typing import Optional
from .ddp import *
import einops
from transformers import AutoConfig
from .Affinity import DSA


def Segformer():
    # id2label = {0: "others"}
    # label2id = {label: id for id, label in id2label.items()}
    # num_labels = len(id2label)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "/home/haipeng/Code/hgf_pretrain/nvidia/segformer-b3-finetuned-ade-512-512",
        ignore_mismatched_sizes=True,
        num_labels=1)
    return model


def SegformerWithMask():
    # id2label = {0: "others"}
    # label2id = {label: id for id, label in id2label.items()}
    # num_labels = len(id2label)
    model = SegformerForSemanticSegmentation.from_pretrained(
        "/home/haipeng/Code/hgf_pretrain/nvidia/segformer-b1-finetuned-ade-512-512",
        ignore_mismatched_sizes=True,
        num_labels=1,
        num_channels=4)
    return model


class PEEDiff(nn.Module):
    def __init__(self,
                 PretrainedSegformer,
                 num_frames,
                 bit_scale=0.1,
                 timesteps=10,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='linear',
                 diffusion="ddim",
                 accumulation=True,
                 ):
        super().__init__()
        # diffusion configs
        self.bit_scale = bit_scale
        self.diffusion_type = diffusion
        self.num_classes = 1
        self.randsteps = randsteps
        self.accumulation = accumulation
        self.sample_range = sample_range
        self.timesteps = timesteps
        self.time_difference = time_difference
        self.x_inp_dim = 64
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.x_inp_dim)
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        # time embeddings for diffusion
        time_dim = 1024  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1
        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

        #  project the cat(noised_gt, bottom_fea)
        self.zip = nn.Sequential(
            nn.BatchNorm2d(self.x_inp_dim * 2),
            nn.Conv2d(self.x_inp_dim * 2, self.x_inp_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.x_inp_dim * 4, self.x_inp_dim, kernel_size=3, stride=1, padding=1)
        )

        ## segformer model
        pre_encoder = PretrainedSegformer.segformer.encoder
        # pre_decoder = PretrainedSegformer.decode_head
        # self.decode_config = AutoConfig
        ## custom encoder
        self.patch_embeddings = pre_encoder.patch_embeddings
        self.block = pre_encoder.block
        self.layer_norm = pre_encoder.layer_norm


        ## affinity
        self.DSA = DSA()

        # past embedding
        self.PretrainSegformerLight = SegformerWithMask()  # past encoder
        # past mix
        self.NowFuseOthers = nn.Sequential(
            nn.BatchNorm2d(self.x_inp_dim * 2),
            nn.Conv2d(self.x_inp_dim * 2, self.x_inp_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.x_inp_dim * 4, self.x_inp_dim, kernel_size=3, stride=1, padding=1)
        )


        self.decoder = SegformerDecodeHeadwTime._from_config(PretrainedSegformer.config)
        self.bce_fct = BCEWithLogitsLoss(reduction="none")

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = einops.repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    def EncodeFrames(self, hidden_states, output_attentions: Optional[bool] = True):
        '''

        :param hidden_states: (b,c,h,w)
        :param output_attentions:
        :return:
        '''
        ##### encoder part #########
        all_hidden_states = ()
        all_self_attentions = () if output_attentions else None

        # a loop for multi-scale feature extract
        for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)

            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)

            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                    idx == len(self.patch_embeddings) - 1  # and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(hidden_states.shape[0], height, width, -1).permute(0, 3, 1,
                                                                                                         2).contiguous()
            all_hidden_states = all_hidden_states + (hidden_states,)

        return all_hidden_states, all_self_attentions

    def ForwardMask(self, frame, label, x):
        '''
        Diffusion forward process
        Follow Bit diffusion and DDP, we transfer the discrete mask (i.e., value only contain int number) into contiguous,
        So we can add noise
        :param frames_flatten: the frames, shape should be b,c,h,w
        :param label_flatten: the masks
        :param x: the bottom layer feature
        :return: noised mask, and a noise level
        '''
        batch, c, h, w = x.size()  # here, the batch should be b*t
        device = x.device
        # label_flatten = labels.flatten(start_dim=0, end_dim=1).contiguous()  # b*t,c,h,w
        # frames_flatten = frames.flatten(start_dim=0, end_dim=1).contiguous()  # b*t,c,h,w

        # reshape gt
        gt_down = resize(label.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(label.dtype)

        # continuous the gt via nn.embed
        gt_down = self.embedding_table(gt_down).squeeze(1).permute(0, 3, 1, 2)
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale

        # sampling time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                      self.sample_range[1])

        # add noise
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(frame, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise
        # cat_fea = torch.cat([x, noised_gt], dim=1)  # b, 64*2,h,w
        return noised_gt, noise_level

    def VisitSpaceTime(self, cur_fea, prev_fea):
        '''

        :param cur_fea: [b,c,h/4,w/4]
        :param prev_fea: [b,t,c,h/4,w/4], t is the num of past frames, agnostic
        :return:
        '''
        # loop for concating all past masks
        # every time concat only onece since the num of past frame is agnostic
        for ii in range(prev_fea.shape[1]):
            past_ii = prev_fea[:, ii]
            cat_past = torch.cat([cur_fea, past_ii], dim=1)
            cat_past = self.NowFuseOthers(cat_past)
        return cat_past

    def forward(self, frames,  labels=None, is_train=True):
        if not is_train:
            return self.ddim_sample(frames)

        assert len(frames.size()) == 5  # must be (b,t,c,h,w)
        if labels is not None:
            assert len(labels.size()) == 5  # must be (b,t,c,h,w)

        bs, num_frame, h, w = frames.shape[0], frames.shape[1], frames.shape[3], frames.shape[4]

        inputs = frames
        inputs_flatten = inputs.flatten(start_dim=0, end_dim=1).contiguous()  # b*t,c,h,w

        ## encoder part, is indendent from affinity and diffusion
        all_hidden_states, _ = self.EncodeFrames(inputs_flatten, False)
        Q4 = all_hidden_states[0].view(bs, num_frame, *all_hidden_states[0].shape[-3:])
        Q8 = all_hidden_states[1].view(bs, num_frame, *all_hidden_states[1].shape[-3:])
        Q16 = all_hidden_states[2].view(bs, num_frame, *all_hidden_states[2].shape[-3:])
        Q32 = all_hidden_states[3].view(bs, num_frame, *all_hidden_states[3].shape[-3:])

        ## affinity part
        Q32 = self.DSA(Q32)


        loss = None
        masks = None
        predicts = None
        # past encode embedding
        for tt in range(num_frame):
            cur_frame = frames[:, tt]  # b,c,h,w
            cur_label = labels[:, tt]
            past_frames = frames[:, :tt]  # b,t,c,h,w
            noised_label, noise_level = self.ForwardMask(cur_frame, cur_label, Q4[:, tt])
            if tt == 0:
                # directly decode as we have no past
                mix_bottom = torch.cat([Q4[:, tt], noised_label], dim=1)  # c:64+64=128
                mix_bottom = self.zip(mix_bottom)  # c:128->64

                input_times = self.time_mlp(noise_level)
                fea_gather = (mix_bottom, Q8[:, tt], Q16[:, tt], Q32[:, tt])
                logits = self.decoder(fea_gather, input_times)

                upsampled_logits = nn.functional.interpolate(
                    logits, scale_factor=4.0, mode="bilinear", align_corners=False)
                mask_cur = torch.sigmoid(upsampled_logits)  #
                masks = mask_cur.unsqueeze(1)
                predicts = upsampled_logits.unsqueeze(1)
            else:
                # 1. concat the past images and pseudo masks
                # 2. encode them via a light-weight encoder
                # 3. extract the bottom features as the guidance
                MaskWithImages = torch.cat([past_frames, masks], dim=2)
                MaskWithImages = MaskWithImages.flatten(start_dim=0, end_dim=1).contiguous()
                PastHiddens = self.PretrainSegformerLight(MaskWithImages, output_hidden_states=True)
                past_bottom = PastHiddens.hidden_states[0]  # b*t, 64,128,128
                past_bottom = past_bottom.view(bs, tt, *past_bottom.shape[-3:])

                # diffusion forward process, we get the corruptted features
                mix_bottom = torch.cat([Q4[:, tt], noised_label], dim=1)
                mix_bottom = self.zip(mix_bottom)
                # add guidance here
                mix_bottom = self.VisitSpaceTime(mix_bottom, past_bottom)

                input_times = self.time_mlp(noise_level)
                fea_gather = (mix_bottom, Q8[:, tt], Q16[:, tt], Q32[:, tt])
                logits = self.decoder(fea_gather, input_times)

                upsampled_logits = nn.functional.interpolate(
                    logits, scale_factor=4.0, mode="bilinear", align_corners=False)
                mask_cur = torch.sigmoid(upsampled_logits)
                masks = torch.cat([masks, mask_cur.unsqueeze(1)], dim=1)
                predicts = torch.cat([predicts, upsampled_logits.unsqueeze(1)], dim=1)

        # loss computation
        predicts_flatten = predicts.flatten(start_dim=0, end_dim=1).contiguous()  # b*t, 1,h,w
        labels_flatten = labels.flatten(start_dim=0, end_dim=1).contiguous()  # b*t, 1,h,w
        valid_mask = ((labels_flatten[:, 0, ...] >= 0)).float()  # keep the mask be valid
        loss_bce = self.bce_fct(predicts_flatten[:, 0], labels_flatten[:, 0].float())
        loss_bce = (loss_bce * valid_mask).mean()
        loss_hinge = lovasz_hinge(predicts_flatten[:, 0], labels_flatten[:, 0].float())
        loss = loss_bce + loss_hinge

        return loss, masks

    @torch.no_grad()
    def ddim_sample_one_frame(self, all_hidden_states, past_bottom=None):
        ## diffusion part
        all_hidden_states = list(all_hidden_states)
        x = all_hidden_states[0]
        b, c, h, w = x.size()
        device = x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)  # list, item has shape of (2,bs)
        x = einops.repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decoder.in_channels[0], h, w), device=device)
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.zip(feat)
            if past_bottom is not None:
                feat = self.VisitSpaceTime(feat, past_bottom)

            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            all_hidden_states[0] = feat

            mask_logit = self.decoder(all_hidden_states, input_times)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next

            if self.accumulation:
                outs.append(mask_logit)
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)

        logits = mask_logit.mean(dim=0, keepdim=True)
        upsampled_logits = nn.functional.interpolate(
            logits, scale_factor=4.0, mode="bilinear", align_corners=False)
        pred = torch.sigmoid(upsampled_logits)  # [b,1,h,w]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        return pred

    @torch.no_grad()
    def ddim_sample(self, frames):
        assert len(frames.size()) == 5  # must be (b,t,c,h,w)
        b, num_frame, h, w = frames.shape[0], frames.shape[1], frames.shape[3], frames.shape[4]
        # by default, b=1, we only provide 1 batch for inference. TODO work leave latter
        inputs = frames
        inputs_flatten = inputs.flatten(start_dim=0, end_dim=1).contiguous()  # b*t,c,h,w

        all_hidden_states, _ = self.EncodeFrames(inputs_flatten, False)
        Q4 = all_hidden_states[0].view(b, num_frame, *all_hidden_states[0].shape[-3:])
        Q8 = all_hidden_states[1].view(b, num_frame, *all_hidden_states[1].shape[-3:])
        Q16 = all_hidden_states[2].view(b, num_frame, *all_hidden_states[2].shape[-3:])
        Q32 = all_hidden_states[3].view(b, num_frame, *all_hidden_states[3].shape[-3:])
        Q32 = self.DSA(Q32)

        for tt in range(num_frame):
            if tt == 0:
                fea_gather = (Q4[:, tt], Q8[:, tt], Q16[:, tt], Q32[:, tt])
                cur_mask = self.ddim_sample_one_frame(fea_gather)
                masks = cur_mask.unsqueeze(1)
            else:
                past_frames = frames[:, :tt]
                MaskWithImages = torch.cat([past_frames, masks], dim=2)
                MaskWithImages = MaskWithImages.flatten(start_dim=0, end_dim=1).contiguous()
                PastHiddens = self.PretrainSegformerLight(MaskWithImages, output_hidden_states=True)
                past_bottom = PastHiddens.hidden_states[0]  # b*t, 64,128,128
                past_bottom = past_bottom.view(b, tt, *past_bottom.shape[-3:])

                # mix_past = self.VisitSpaceTime(Q4[:, tt], past_bottom)
                fea_gather = (Q4[:, tt], Q8[:, tt], Q16[:, tt], Q32[:, tt])
                cur_mask = self.ddim_sample_one_frame(fea_gather, past_bottom)
                masks = torch.cat([masks, cur_mask.unsqueeze(1)], dim=1)

        return _, masks
