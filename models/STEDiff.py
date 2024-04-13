from transformers import SegformerForSemanticSegmentation, SegformerPreTrainedModel, SegformerConfig
import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from .loss_func import lovasz_hinge
from typing import Optional
from .ddp import *
import einops
from .Affinity import DSA
from .SBBA import SBBA


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


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class AuxHead(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.pseudo_mask_decoder = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)
        self.edge_decoder = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        self.config = config

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.pseudo_mask_decoder(hidden_states)
        edge = self.edge_decoder(hidden_states)

        return logits, edge


class STAN(nn.Module):
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

        self.NowFuseOthers = nn.Sequential(
            nn.BatchNorm2d(self.x_inp_dim * 5),
            nn.Conv2d(self.x_inp_dim * 5, self.x_inp_dim * 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.x_inp_dim * 1, self.x_inp_dim, kernel_size=3, stride=1, padding=1)
        )

        ## segformer model
        self.pretrained_encoder = PretrainedSegformer.segformer.encoder

        ## custom encoder
        self.patch_embeddings = self.pretrained_encoder.patch_embeddings
        self.block = self.pretrained_encoder.block
        self.layer_norm = self.pretrained_encoder.layer_norm

        ## DSA
        self.DSA = DSA()

        ## auxiliary head
        self.aux_head = AuxHead._from_config(PretrainedSegformer.config)

        ## SBAA
        self.SBAA_Q4 = SBBA(dim=64)
        self.SBAA_Q8 = SBBA(dim=128)
        self.SBAA_Q16 = SBBA(dim=320)
        self.SBAA_Q32 = SBBA(dim=512)

        # guidance encoder
        self.PretrainSegformerLight = SegformerWithMask()

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

        :param bottom_fea: [b,64,h/4,w/4]
        :param prev_mask: [b,t,64,h/4,w/4]
        :return:
        '''
        # loop for concating all past masks
        # every time concat only one
        time_line = torch.cat([cur_fea, prev_fea[:, 0], prev_fea[:, 1],
                               prev_fea[:, 2], prev_fea[:, 3]], dim=1)
        time_line = self.NowFuseOthers(time_line) + cur_fea  # residual connection
        '''for ii in range(prev_fea.shape[1]):
            past_ii = prev_fea[:, ii]
            cat_past = torch.cat([cur_fea, past_ii], dim=1)
            cat_past = self.NowFuseOthers(cat_past)'''
        return time_line

    def forward(self, frames, labels=None, labels_boundary=None, is_train=True):
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

        ## affinity part, only apply for Q32
        Q32 = self.DSA(Q32)

        aggregated_features = (Q4.flatten(start_dim=0, end_dim=1).contiguous(),
                               Q8.flatten(start_dim=0, end_dim=1).contiguous(),
                               Q16.flatten(start_dim=0, end_dim=1).contiguous(),
                               Q32.flatten(start_dim=0, end_dim=1).contiguous())

        pseudo_logits, boundary_logits = self.aux_head(aggregated_features)

        pseudo_logits = pseudo_logits.view(bs, num_frame, *pseudo_logits.shape[-2:])
        pseudo_logits = nn.functional.interpolate(
            pseudo_logits, scale_factor=4.0, mode="bilinear", align_corners=False)
        pseudo_masks = F.sigmoid(pseudo_logits)
        pseudo_masks = torch.unsqueeze(pseudo_masks, 2)
        boundary_logits = boundary_logits.view(bs, num_frame, *boundary_logits.shape[-2:])
        boundary_logits = nn.functional.interpolate(
            boundary_logits, scale_factor=4.0, mode="bilinear", align_corners=False)
        boundary_mask = F.sigmoid(boundary_logits)
        boundary_mask = torch.unsqueeze(boundary_mask, 2)

        Q4 = self.SBAA_Q4(Q4.flatten(start_dim=0, end_dim=1).contiguous(),
                          pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                          boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(bs, num_frame, *all_hidden_states[0].shape[-3:])
        Q8 = self.SBAA_Q8(Q8.flatten(start_dim=0, end_dim=1).contiguous(),
                          pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                          boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(bs, num_frame, *all_hidden_states[1].shape[-3:])
        Q16 = self.SBAA_Q16(Q16.flatten(start_dim=0, end_dim=1).contiguous(),
                            pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                            boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(bs, num_frame, *all_hidden_states[2].shape[-3:])
        Q32 = self.SBAA_Q32(Q32.flatten(start_dim=0, end_dim=1).contiguous(),
                            pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                            boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(bs, num_frame, *all_hidden_states[3].shape[-3:])

        loss = None
        masks = None
        predicts = None
        coarse_masks = pseudo_masks

        # store the noised config so we can apply the same noise
        noised_label_list = []
        noise_level_list = []

        # apply space-time embedding with all other guidance
        for tt in range(num_frame):
            cur_frame = frames[:, tt]  # b,c,h,w
            cur_label = labels[:, tt]

            noised_label, noise_level = self.ForwardMask(cur_frame, cur_label, Q4[:, tt])
            noised_label_list.append(noised_label)
            noise_level_list.append(noise_level)
            # corrupted features
            mix_bottom = torch.cat([Q4[:, tt], noised_label], dim=1)
            mix_bottom = self.zip(mix_bottom)

            # add guidance
            other_frames = torch.cat([frames[:, :tt, ...], frames[:, tt + 1:, ...]], dim=1)  # b, num_frame-1, c,h,w
            other_raw_masks = torch.cat([coarse_masks[:, :tt, ...], coarse_masks[:, tt + 1:, ...]], dim=1)
            MaskWithImages = torch.cat([other_frames, other_raw_masks], dim=2)
            MaskWithImages = MaskWithImages.flatten(start_dim=0, end_dim=1).contiguous()
            OtherHiddens = self.PretrainSegformerLight(MaskWithImages, output_hidden_states=True)
            other_bottom = OtherHiddens.hidden_states[0]  # b*t, 64,128,128
            other_bottom = other_bottom.view(bs, num_frame - 1, *other_bottom.shape[-3:])
            mix_bottom = self.VisitSpaceTime(mix_bottom, other_bottom)  # Space-Time Embedding

            input_times = self.time_mlp(noise_level)
            fea_gather = (mix_bottom, Q8[:, tt], Q16[:, tt], Q32[:, tt])
            logits = self.decoder(fea_gather, input_times)
            upsampled_logits = nn.functional.interpolate(
                logits, scale_factor=4.0, mode="bilinear", align_corners=False)
            mask_cur = F.sigmoid(upsampled_logits)

            if tt == 0:
                masks = mask_cur.unsqueeze(1)
                predicts = upsampled_logits.unsqueeze(1)
            else:
                masks = torch.cat([masks, mask_cur.unsqueeze(1)], dim=1)
                predicts = torch.cat([predicts, upsampled_logits.unsqueeze(1)], dim=1)

        ## Loss part
        predicts_flatten = predicts.flatten(start_dim=0, end_dim=1).contiguous()  # b*t, 1,h,w
        pseudo_logits_flatten = pseudo_logits.flatten(start_dim=0, end_dim=1).contiguous()  # b*t, 1,h,w
        pseudo_logits_flatten = torch.unsqueeze(pseudo_logits_flatten, dim=1)
        boundary_logits_flatten = boundary_logits.flatten(start_dim=0, end_dim=1).contiguous()  # b*t, 1,h,w
        boundary_logits_flatten = torch.unsqueeze(boundary_logits_flatten, dim=1)

        labels_flatten = labels.flatten(start_dim=0, end_dim=1).contiguous()  # b*t, 1,h,w
        labels_boundary_flatten = labels_boundary.flatten(start_dim=0, end_dim=1).contiguous()  # b*t, 1,h,w

        valid_mask = ((labels_flatten[:, 0, ...] >= 0)).float()  # keep the mask be valid
        valid_boundary = ((labels_boundary_flatten[:, 0, ...] >= 0)).float()  # keep the mask be valid

        loss_bce = self.bce_fct(predicts_flatten[:, 0], labels_flatten[:, 0].float())
        loss_bce = (loss_bce * valid_mask).mean()
        loss_hinge = lovasz_hinge(predicts_flatten[:, 0], labels_flatten[:, 0].float())

        loss_raw = self.bce_fct(pseudo_logits_flatten[:, 0], labels_flatten[:, 0].float())
        loss_raw = (loss_raw * valid_mask).mean()

        loss_boundary = self.bce_fct(boundary_logits_flatten[:, 0], labels_boundary_flatten[:, 0].float())
        loss_boundary = (loss_boundary * valid_boundary).mean()

        loss = loss_bce + loss_hinge + 0.5 * loss_raw + loss_boundary

        return loss, masks, pseudo_masks, boundary_mask

    @torch.no_grad()
    def ddim_sample_one_frame(self, all_hidden_states, other_bottom=None):
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
            feat = self.VisitSpaceTime(feat, other_bottom)

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

        aggregated_features = (Q4.flatten(start_dim=0, end_dim=1).contiguous(),
                               Q8.flatten(start_dim=0, end_dim=1).contiguous(),
                               Q16.flatten(start_dim=0, end_dim=1).contiguous(),
                               Q32.flatten(start_dim=0, end_dim=1).contiguous())
        pseudo_logits, boundary_logits = self.aux_head(aggregated_features)

        pseudo_logits = pseudo_logits.view(b, num_frame, *pseudo_logits.shape[-2:])
        pseudo_logits = nn.functional.interpolate(
            pseudo_logits, scale_factor=4.0, mode="bilinear", align_corners=False)
        pseudo_masks = F.sigmoid(pseudo_logits)
        pseudo_masks = torch.unsqueeze(pseudo_masks, 2)
        boundary_logits = boundary_logits.view(b, num_frame, *boundary_logits.shape[-2:])
        boundary_logits = nn.functional.interpolate(
            boundary_logits, scale_factor=4.0, mode="bilinear", align_corners=False)
        boundary_mask = F.sigmoid(boundary_logits)
        boundary_mask = torch.unsqueeze(boundary_mask, 2)

        Q4 = self.SBAA_Q4(Q4.flatten(start_dim=0, end_dim=1).contiguous(),
                          pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                          boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(b, num_frame, *all_hidden_states[0].shape[-3:])
        Q8 = self.SBAA_Q8(Q8.flatten(start_dim=0, end_dim=1).contiguous(),
                          pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                          boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(b, num_frame, *all_hidden_states[1].shape[-3:])
        Q16 = self.SBAA_Q16(Q16.flatten(start_dim=0, end_dim=1).contiguous(),
                            pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                            boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(b, num_frame, *all_hidden_states[2].shape[-3:])
        Q32 = self.SBAA_Q32(Q32.flatten(start_dim=0, end_dim=1).contiguous(),
                            pseudo_masks.flatten(start_dim=0, end_dim=1).contiguous(),
                            boundary_mask.flatten(start_dim=0, end_dim=1).contiguous()).view(b, num_frame, *all_hidden_states[3].shape[-3:])

        '''pseudo_logits = self.pretrained_decoder(aggregated_features)
        pseudo_logits = pseudo_logits.view(b, num_frame, *pseudo_logits.shape[-2:])
        pseudo_logits = nn.functional.interpolate(
            pseudo_logits, scale_factor=4.0, mode="bilinear", align_corners=False)
        pseudo_masks = F.sigmoid(pseudo_logits)
        pseudo_masks = torch.unsqueeze(pseudo_masks, 2)'''

        masks = None
        coarse_masks = pseudo_masks

        for tt in range(num_frame):
            other_frames = torch.cat([frames[:, :tt, ...], frames[:, tt + 1:, ...]], dim=1)  # b, num_frame-1, c,h,w
            other_raw_masks = torch.cat([coarse_masks[:, :tt, ...], coarse_masks[:, tt + 1:, ...]], dim=1)

            MaskWithImages = torch.cat([other_frames, other_raw_masks], dim=2)
            MaskWithImages = MaskWithImages.flatten(start_dim=0, end_dim=1).contiguous()
            OtherHiddens = self.PretrainSegformerLight(MaskWithImages, output_hidden_states=True)
            other_bottom = OtherHiddens.hidden_states[0]  # b*t, 64,128,128
            other_bottom = other_bottom.view(b, num_frame - 1, *other_bottom.shape[-3:])
            # mix_other = self.VisitSpaceTime(Q4[:, tt], other_bottom)  # Space-Time Embedding

            fea_gather = (Q4[:, tt], Q8[:, tt], Q16[:, tt], Q32[:, tt])
            cur_mask = self.ddim_sample_one_frame(fea_gather, other_bottom)
            if tt == 0:
                masks = cur_mask.unsqueeze(1)
            else:
                masks = torch.cat([masks, cur_mask.unsqueeze(1)], dim=1)
        return _, masks
