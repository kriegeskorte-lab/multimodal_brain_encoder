import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from metric_logger import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_transformer
from models.custom_transformer import build_custom_transformer
from models.position_encoding import build_position_encoding

from transformers import BertModel, GPT2Model
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import open_clip
from transformers import WhisperProcessor, WhisperModel
from transformers import TimesformerModel, AutoModel, AutoConfig, AutoModelForCausalLM


class brain_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.sub = args.sub

        self.lr_backbone = args.lr_backbone

       
        self.return_interm = args.return_interm
        self.encoder_arch = args.encoder_arch
        self.modality = args.modality
        self.num_frames = args.num_frames

        self.backbone_arch = args.visual_bb
        self.visual_bb = args.visual_bb
        ### backbone_arch for feature exraction

        self.second_visual_bb = args.second_visual_bb
            
        self.text_bb = args.text_bb
        self.second_text_bb = args.second_text_bb
        self.third_text_bb = args.third_text_bb

        self.audio_bb = args.audio_bb
        self.second_audio_bb = args.second_audio_bb

        self.combine_modality = False
        self.index_trial = args.index_trial

        self.video_bb = args.video_bb #'VideoMAEv2' #'timesformer'

        # number of brain areas
        self.num_queries = args.num_queries
        if args.readout_res == "voxels":
            self.num_voxels = args.num_voxels

        # self.audio_processor = ASTProcessor.from_pretrained("facebook/ast-base")
        # self.audio_model = ASTModel.from_pretrained("facebook/ast-base")
        token_dim = 0
        ### Brain encoding model
        if 'transformer' in args.encoder_arch:
            if args.encoder_arch == 'transformer':

                self.hidden_dim = 768 # self.transformer.d_model
                self.linear_feature_dim  = self.hidden_dim

                if 'visual' in args.modality:
                    if 'dino' in self.visual_bb:
                        self.visual_backbone = build_backbone(args, backbone_arch=self.visual_bb)
                        self.visual_transformer = build_transformer(args)
                        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                        token_dim += self.linear_feature_dim

                        self.visual_hidden_dim = 768
                        self.visual_time_embed = nn.Embedding(self.num_frames, self.hidden_dim)

                    if 'clip' in self.second_visual_bb:
                        self.second_visual_backbone = build_backbone(args, backbone_arch=self.second_visual_bb)
                        hidden_dim = self.hidden_dim 
                        if self.lr_backbone == 0:
                            for param in self.second_visual_backbone.parameters():
                                param.requires_grad = False

                        if self.second_visual_bb == 'clip_bigG':
                            hidden_dim = 1280 

                        self.second_visual_transformer = build_transformer(args, token_emb_dim=hidden_dim)
                        self.second_query_embed = nn.Embedding(self.num_queries, hidden_dim)
                        self.second_visual_time_embed = nn.Embedding(self.num_frames, hidden_dim)
                        token_dim += hidden_dim

                        self.second_visual_hidden_dim = hidden_dim

                    # Load video model
                    if self.video_bb == 'timesformer':
                        self.video_backbone = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
                        self.video_backbone.eval()
                        for param in self.video_backbone.parameters():
                            param.requires_grad = False
                        self.video_transformer = build_transformer(args)
                        self.video_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                        self.video_pos_embed = torch.zeros(1, self.hidden_dim, 16, 24).cuda() 
                        token_dim += self.linear_feature_dim
                    
                    elif self.video_bb == 'InternVideo':
                        # model setting
                        video_model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'

                        self.video_backbone = AutoModel.from_pretrained(video_model_path, trust_remote_code=True).half().cuda().to(torch.bfloat16)
                        self.video_backbone.eval()
                        for param in self.video_backbone.parameters():
                            param.requires_grad = False

                        self.video_transformer = build_transformer(args)
                        self.video_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                        video_token_dim = 4096
                        self.video_input_proj = nn.Conv2d(video_token_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0)
                        # TODO hard coding the video pos embed size
                        self.video_pos_embed = torch.zeros(1, self.hidden_dim, 320).cuda()
                        token_dim += self.linear_feature_dim


                    elif self.video_bb == 'VideoMAEv2':
                        config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
                        config.output_hidden_states = True
                        self.video_backbone = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=config, trust_remote_code=True)
                        for param in self.video_backbone.parameters():
                            param.requires_grad = False
                        token_dim += self.linear_feature_dim

                if 'audio' in args.modality:

                    if self.audio_bb == 'Wav2Vec':
                        self.audio_processor = self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                        self.audio_backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                        for param in self.audio_backbone.parameters():
                            param.requires_grad = False
                        self.audio_transformer = build_transformer(args)
                        self.audio_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                        self.audio_time_embed = nn.Embedding(1117, self.hidden_dim)
                        token_dim += self.linear_feature_dim

                    if 'whisper' in self.second_audio_bb:
                        if self.second_audio_bb == 'whisper':
                            self.second_audio_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                            self.second_audio_backbone = WhisperModel.from_pretrained("openai/whisper-small")
                            hidden_dim = self.hidden_dim
                            token_dim += hidden_dim
                        elif self.second_audio_bb == 'whisper_large_v2':
                            self.second_audio_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
                            self.second_audio_backbone = WhisperModel.from_pretrained("openai/whisper-large-v2")
                            hidden_dim = 1280
                            token_dim += hidden_dim

                        for param in self.second_audio_backbone.parameters():
                            param.requires_grad = False
                        self.second_audio_transformer = build_transformer(args, token_emb_dim=hidden_dim)
                        self.second_audio_query_embed = nn.Embedding(self.num_queries, hidden_dim)
                        self.second_audio_time_embed = nn.Embedding(1500, hidden_dim)
                        

                if 'text' in args.modality:

                    if 'bert' in self.text_bb:
                        if self.text_bb == 'bert':
                            self.text_backbone = BertModel.from_pretrained("bert-base-uncased")
                            hidden_dim = self.hidden_dim
                        elif self.text_bb == 'deberta_v2_xlarge':
                            model_name = "microsoft/deberta-v2-xlarge"
                            self.text_backbone = AutoModel.from_pretrained(model_name)
                            self.text_backbone.eval()
                            hidden_dim = 1536

                        #self.lang_model = GPT2Model.from_pretrained("gpt2")
                        for param in self.text_backbone.parameters():
                            param.requires_grad = False
                        self.text_transformer = build_transformer(args, token_emb_dim=hidden_dim)
                        self.text_query_embed = nn.Embedding(self.num_queries, hidden_dim)
                        self.text_time_embed = nn.Embedding(128, hidden_dim)
                        token_dim += hidden_dim
            

                    if 'clip' in self.second_text_bb:
                        self.second_text_backbone, _, _ =  open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
                        if self.lr_backbone == 0:
                            for param in self.second_text_backbone.parameters():
                                param.requires_grad = False

                        self.second_text_transformer = build_transformer(args)
                        self.second_text_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                        self.second_text_time_embed = nn.Embedding(77, self.hidden_dim)
                        token_dim += self.linear_feature_dim
            

                    if self.third_text_bb == 'llama':
                        model_name = "meta-llama/Meta-Llama-3-8B"

                        self.third_text_backbone = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype=torch.float16,
                            device_map=None  # We'll move manually below
                        )
                        self.third_text_backbone = self.third_text_backbone.cuda()  # Manually move to GPU

                        self.third_text_backbone.config.output_hidden_states = True
                        self.third_text_backbone.eval()

                        for param in self.third_text_backbone.parameters():
                            param.requires_grad = False

                        self.third_text_transformer = build_transformer(args)
                        self.third_text_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

                        text_token_dim = 4096
                        self.text_input_proj = nn.Conv2d(text_token_dim, self.hidden_dim, kernel_size=1, stride=1, padding=0)
                        self.third_text_time_embed = nn.Embedding(128, self.hidden_dim)
                        token_dim += self.linear_feature_dim

                if self.combine_modality == True:
                    self.combined_transformer = build_transformer(args)
                    self.combined_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                    token_dim += self.linear_feature_dim

                    self.second_combined_transformer = build_transformer(args)
                    self.second_combined_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                    token_dim += self.linear_feature_dim

            elif self.encoder_arch == 'custom_transformer':
                self.transformer = build_custom_transformer(args)

            if ('resnet' in self.backbone_arch):
                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=1)
        
        elif self.encoder_arch == 'spatial_feature':

            #TODO hard coding the map size for now but fix it
            self.map_size = 31
            self.spatial_embed = nn.Embedding(self.num_queries, self.map_size*self.map_size)
            self.linear_feature_dim = self.backbone_model.num_channels

            self.downsize = False
            if self.downsize: 
                self.hidden_dim = 256
                if 'resnet' in self.backbone_arch:
                    stride=1
                    self.map_size = 11
                else:
                    stride=3
                    self.map_size = 11

                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=3, stride=stride, padding=1)
                    
                # for each roi, learn a spatial map
                self.spatial_embed = nn.Embedding(self.num_queries, self.map_size*self.map_size)
                self.linear_feature_dim = self.hidden_dim

        elif self.encoder_arch == 'linear':
            #TODO hard  coding the map size and hidden dimention for now but fix it
            # using conv to make the input smaller for linear layer
            self.hidden_dim = 256
            if 'resnet' in self.backbone_arch:
                stride=1
                self.map_size = 11
            else:
                stride=3
                self.map_size = 11

            #if 'dino' in self.backbone_arch:
            self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=3, stride=stride, padding=1)
                
            # if ('resnet' in self.backbone_arch):
            #     self.input_proj = nn.AdaptiveAvgPool2d(1)
            self.linear_feature_dim = self.hidden_dim*self.map_size*self.map_size


        if 'visual' in args.modality:
            # time embedding # TODO could be learnable or fixed
            
            if self.video_bb == 'timesformer':
                self.video_pos_embed = nn.Embedding(3137, self.hidden_dim)
            elif self.video_bb == 'InternVideo':
                self.video_pos_embed = nn.Embedding(320, self.hidden_dim)
            

        if self.index_trial == 1:
            self.position_embedding = build_position_encoding(args.position_embedding, 32)
            b = 1 # brain_data.shape[0]
            x = torch.ones(b, self.hidden_dim, 1000, 1).cuda() # TODO # frames
            x = nested_tensor_from_tensor_list(tuple(x))
            index_embedding = self.position_embedding(x)
            self.index_embedding = index_embedding.squeeze().flatten(1).permute(1, 0) #[1000,64]
        
            token_dim += 64


        self.readout_res = args.readout_res

        self.readout_mapping = args.readout_mapping
        if self.readout_res == 'parcels': # or self.readout_res == 'all_parcels':

            if 'transformer' in self.readout_mapping:
                self.readout_transformer = build_transformer(args)
                self.readout_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                self.readout_pos_embed = nn.Embedding(6000, self.hidden_dim)
                token_dim += self.hidden_dim

            self.vs = 1000
            if 'linear' in self.readout_mapping:
                if self.sub == 0:
                    self.lin_embed_1 = nn.Sequential(
                        nn.Linear(token_dim, self.vs),
                    )
                    self.lin_embed_2 = nn.Sequential(
                        nn.Linear(token_dim, self.vs),
                    )
                    self.lin_embed_3 = nn.Sequential(
                        nn.Linear(token_dim, self.vs),
                    )
                    self.lin_embed_5 = nn.Sequential(
                        nn.Linear(token_dim, self.vs),
                    )
                else:
                    self.lin_embed = nn.Sequential(
                        nn.Linear(token_dim, self.vs),
                    )


        elif self.readout_res == "voxels":
            self.lin_embed = nn.Sequential(
                nn.Linear(
                    token_dim,
                    self.num_voxels,
                ),
            )
            self.parcel_mask = torch.zeros(
                self.num_queries,
                args.num_voxels,
            ).to(args.device)
            masked_parcellation = torch.from_numpy(
                args.masked_parcellation.astype(int)
            ).to(args.device)
            for i in range(1,1001): # torch.unique(masked_parcellation):
                parcel_idxs = torch.where(masked_parcellation == i)[0]
                self.parcel_mask[i-1, parcel_idxs] = 1

            # elif self.readout_mapping == 'linear_2':
            #     bottleneck_dim = self.vs // 2
            #     self.lin_embed_1 = nn.Sequential(
            #         nn.Linear(token_dim, bottleneck_dim),
            #         nn.ReLU(),
            #         nn.Linear(bottleneck_dim, self.vs),
            #     )
            #     self.lin_embed_2 = nn.Sequential(
            #         nn.Linear(token_dim, bottleneck_dim),
            #         nn.ReLU(),
            #         nn.Linear(bottleneck_dim, self.vs),
            #     )
            #     self.lin_embed_3 = nn.Sequential(
            #         nn.Linear(token_dim, bottleneck_dim),
            #         nn.ReLU(),
            #         nn.Linear(bottleneck_dim, self.vs),
            #     )
            #     self.lin_embed_5 = nn.Sequential(
            #         nn.Linear(token_dim, bottleneck_dim),
            #         nn.ReLU(),
            #         nn.Linear(bottleneck_dim, self.vs),
            #     )

            

            # self.relu = nn.ReLU()
            # self.lin_embed_2 = nn.Sequential(
            #     nn.Linear(self.vs//2, self.vs),
            # )
        # elif self.readout_res == 'hemis':
        #     self.vs = 500
        #     self.lh_embed = nn.Sequential(
        #         nn.Linear(token_dim, self.vs),
        #     )
                        
        #     self.rh_embed = nn.Sequential(
        #         nn.Linear(token_dim, self.vs),
        #     )
        
        # self.rh_embed = nn.Sequential(
        #     nn.Linear(self.linear_feature_dim, 1000),
        # )
            

    def forward(self, samples: NestedTensor):

        #### visual input
        if 'visual' in self.modality:
            if self.visual_bb == 'dinov2_q':
                frames = samples['visual'] #[:,8].cuda()
                
                b, f, c, h, w = frames.shape
                frames = frames.view(b*f, c, h, w).cuda()

                if isinstance(frames, (list, torch.Tensor)):
                    frames = nested_tensor_from_tensor_list(frames)

                # if self.backbone_arch:
                # if self.lr_backbone == 0:
                with torch.no_grad():
                    features, pos = self.visual_backbone(frames)
                # else:
                #     features, pos = self.backbone_model(frames)

                input_proj_src, mask = features[-1].decompose()

                pos_embed = pos[-1]
                _,_,h,w = pos_embed.shape


                # print('frames:', frames.shape)
                # print('input_proj_src.shape:', input_proj_src.shape)
                # print('mask.shape:', mask.shape)
                # print('pos_embed.shape:', pos_embed.shape)

                # input_proj_src.shape: torch.Size([20, 768, 16, 24])
                # mask.shape: torch.Size([20, 16, 24])
                # pos_embed.shape: torch.Size([20, 768, 16, 24])
                # input_proj_audio: torch.Size([1, 768, 482, 1])

                input_proj_src = input_proj_src.view(b, f, self.hidden_dim, h, w).permute(0, 2, 1, 3, 4)
                pos_embed = pos_embed.view(b, f, self.hidden_dim, h, w)
                mask = mask.view(b, f, h, w) #.permute(0, 2, 1)

                visual_time_embed = self.visual_time_embed.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
                visual_time_embed = visual_time_embed.repeat(b, 1, 1, h, w)
                pos_embed = pos_embed + visual_time_embed
                pos_embed = pos_embed.permute(0, 2, 1, 3, 4)

            if 'clip' in self.second_visual_bb:

                frames = samples['visual'] #[:,8].cuda()

                b, f, c, h, w = frames.shape
                frames = frames.view(b*f, c, h, w).cuda()
                frames = F.interpolate(frames, size=(224, 224), mode='bilinear', align_corners=False)

                if isinstance(frames, (list, torch.Tensor)):
                    frames = nested_tensor_from_tensor_list(frames)

                # if self.backbone_arch:
                if self.lr_backbone == 0:
                    with torch.no_grad():
                        features, pos = self.second_visual_backbone(frames)
                else:
                    features, pos = self.second_visual_backbone(frames)

                second_visual_input, second_visual_mask = features[-1].decompose()

                second_pos_embed = pos[-1]
                _,_,second_h,second_w = second_pos_embed.shape

                second_visual_input = second_visual_input.view(b, f, self.second_visual_hidden_dim, second_h, second_w).permute(0, 2, 1, 3, 4)
                second_pos_embed = second_pos_embed.view(b, f, self.second_visual_hidden_dim, second_h, second_w)
                second_visual_mask = second_visual_mask.view(b, f, second_h, second_w) #.permute(0, 2, 1)

                visual_time_embed = self.second_visual_time_embed.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
                visual_time_embed = visual_time_embed.repeat(b, 1, 1, second_h, second_w)
                second_pos_embed = second_pos_embed + visual_time_embed
                second_pos_embed = second_pos_embed.permute(0, 2, 1, 3, 4)


            ### video model 
            if self.video_bb == 'timesformer':
                frames = samples['video'] #[:,8].cuda()
                pixel_values = frames.squeeze(1).cuda()
                #pixel_values = frames.permute(0, 2, 1, 3, 4).cuda()  # Change to [batch_size, channels, frames, height, width]
                b, f, c, h, w = pixel_values.shape

                # Forward pass
                with torch.no_grad():
                    outputs = self.video_backbone(pixel_values=pixel_values)
                    video_embeddings = outputs.last_hidden_state #torch.Size([b, 3137, 768])

                seq = video_embeddings.shape[1]

                input_proj_video = video_embeddings.unsqueeze(-1).permute(0, 2, 1, 3)
                video_pos_embed = self.video_pos_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                video_pos_embed = video_pos_embed.permute(0, 2, 1, 3)
                #video_pos_embed= torch.zeros_like(input_proj_video).cuda() #.permute(0, 2, 1, 3)
                video_mask = torch.zeros(b, seq, 1).cuda()

                # pixel_values: torch.Size([4, 16, 3, 224, 224])
                # video token embeddings: torch.Size([4, 3137, 768])
                # input_proj_video: torch.Size([4, 768, 3137, 1])

            if self.video_bb == 'VideoMAEv2':
                frames = samples['video'].squeeze(1)
                
                pixel_values = frames.permute(0, 2, 1, 3, 4).cuda()

                b, c, f, h, w = pixel_values.shape

                # Forward pass
                with torch.no_grad():
                    video_embeddings = self.video_backbone(pixel_values=pixel_values) # [1, 768]

            elif self.video_bb == 'InternVideo':
                #TODO chech that the batch size is 1    
                pixel_values = samples['video'] #[:,8].cuda() #torch.Size([1, 20, 3, 224, 224])
                pixel_values = pixel_values.squeeze(0)
                with torch.no_grad():
                    pixel_values = pixel_values.to(torch.bfloat16).cuda()
                    video_embeddings = self.video_backbone.extract_feature(pixel_values)
                    # torch.Size([20, 16, 4096])
                    num_frames, num_patches, video_token_dim = video_embeddings.shape
                    #print(f"video_embeddings shape: {video_embeddings.shape}")

                video_embeddings = video_embeddings.reshape(1, num_frames*num_patches, video_token_dim, 1)
                video_embeddings = video_embeddings.permute(0, 2, 1, 3) # [1, 4096, 320, 1]
                input_proj_video = self.video_input_proj(video_embeddings.to(torch.float32)) #torch.Size([1, 768, 320, 1])
                b, _, seq, _ = input_proj_video.shape
                #torch.zeros(1, self.hidden_dim, 320)
                #video_pos_embed = self.video_pos_embed.unsqueeze(-1).cuda()
                video_pos_embed = self.video_pos_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                video_pos_embed = video_pos_embed.permute(0, 2, 1, 3)
                #video_pos_embed = video_pos_embed.permute(0, 2, 1, 3)
                video_mask = torch.zeros(b, seq, 1).cuda()

        #### audio inputw
        if 'audio' in self.modality:

            if self.audio_bb == 'Wav2Vec':
                audio = samples['audio'].cuda() 
                # sr = samples['sr'].cuda()[0]
                b, _= audio.shape

                audio_input = self.audio_processor(
                    audio, sampling_rate=samples['sr'][0], return_tensors="pt", padding=True
                )
                with torch.no_grad():
                    input_audio = self.audio_backbone(audio).last_hidden_state

                b, seq, _ = input_audio.shape

                # input_proj_audio = self.audio_proj(input_audio.unsqueeze(-1))
                input_proj_audio= input_audio.unsqueeze(-1).permute(0, 2, 1, 3)
                audio_time_embed = self.audio_time_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                audio_pos_embed = audio_time_embed.permute(0, 2, 1, 3)
                audio_mask = torch.zeros(b, seq, 1).cuda()

            if 'whisper' in self.second_audio_bb:

                audio = samples['audio'].cuda() 
                # sr = samples['sr'].cuda()[0]
                b, _= audio.shape

                audio_list = audio.tolist()

                with torch.no_grad():
                    audio_input = self.second_audio_processor(audio_list, return_tensors="pt", sampling_rate=16000)
                    audio_input = {key: value.cuda() for key, value in audio_input.items()}
                    outputs = self.second_audio_backbone.encoder(**audio_input)
                    second_input_audio = outputs.last_hidden_state

                #b, seq, _ = input_audio.shape

                # input_proj_audio = self.audio_proj(input_audio.unsqueeze(-1))
                second_input_proj_audio= second_input_audio.unsqueeze(-1).permute(0, 2, 1, 3)
                second_audio_time_embed = self.second_audio_time_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                second_audio_pos_embed = second_audio_time_embed.permute(0, 2, 1, 3)
                second_audio_mask = torch.zeros(b, 1500, 1).cuda()

            #print('input_proj_audio:', input_proj_audio.shape)

            # input_proj_audio: torch.Size([1, 768, 482, 1])
            # audio_pos_embed: torch.Size([1, 768, 482, 1])

            # Preprocess the audio
            # audio = self.audio_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

        #### text input
        if 'text' in self.modality:

            if 'bert' in self.text_bb:
                text = samples['text']
                text = {key: value.squeeze(1).cuda() for key, value in text.items()}
        

                # Extract token embeddings
                with torch.no_grad():
                    text_embeddings = self.text_backbone(**text)

                text_embeddings = text_embeddings.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
                #text_embeddings = text_embeddings['pooler_output'].unsqueeze(1)

                input_proj_text = text_embeddings.permute(0, 2, 1).unsqueeze(-1)
                text_time_embed = self.text_time_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                text_pos_embed = text_time_embed.permute(0, 2, 1, 3)
                text_mask = torch.zeros(text_embeddings.shape[0], 128, 1).cuda()

            if self.second_text_bb == 'clip':
                text = samples['text_clip'].cuda()
                #text = {key: value.squeeze(1).cuda() for key, value in text.items()}
                # Extract token embeddings
                if self.lr_backbone == 0:
                    with torch.no_grad():
                        second_text_embeddings = self.second_text_backbone.token_embedding(text) 
                else:
                    second_text_embeddings = self.second_text_backbone.token_embedding(text) 

                second_input_proj_text = second_text_embeddings.permute(0, 3, 2, 1)
                second_text_time_embed = self.second_text_time_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                secondd_text_pos_embed = second_text_time_embed.permute(0, 2, 1, 3)
                second_text_mask = torch.zeros(second_text_embeddings.shape[0], 77, 1).cuda()

            if self.third_text_bb == 'llama':
                text = samples['text_llama']
                text = {key: value.squeeze(1).cuda() for key, value in text.items()}

                # Extract token embeddings
                with torch.no_grad():
                    outputs = self.third_text_backbone(**text)
                    hidden_states = outputs.hidden_states
                    third_text_embeddings = hidden_states[-1] # # [1, seq_len, 4096]

                b, seq_len, text_token_dim = third_text_embeddings.shape
                third_text_embeddings = third_text_embeddings.reshape(1, seq_len, text_token_dim, 1)
                third_text_embeddings = third_text_embeddings.permute(0, 2, 1, 3) # [1, 4096, 120, 1]
                third_input_proj_text = self.text_input_proj(third_text_embeddings.to(torch.float32)) #torch.Size([1, 768, 120, 1])

                third_text_time_embed = self.third_text_time_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                third_text_pos_embed = third_text_time_embed.permute(0, 2, 1, 3)
                third_text_mask = torch.zeros(b, 128, 1).cuda()

            #print('input_proj_text:', input_proj_text.shape)

        if self.encoder_arch == 'transformer':
            
        # if backbone is resnet, apply 1x1 conv to project the feature to the transformer dimension
            # if 'resnet' in self.backbone_arch:
            #     input_proj_src = self.input_proj(input_proj_src)

            output_tokens = []

            ## visual transformer
            if 'visual' in self.modality:
                if 'dino' in self.visual_bb:
                    hs = self.visual_transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
                    visual_tokens = hs[-1]
                    output_tokens.append(visual_tokens)

                if 'clip' in self.second_visual_bb:
                    hs = self.second_visual_transformer(second_visual_input, second_visual_mask, self.second_query_embed.weight, second_pos_embed, self.return_interm)
                    second_visual_tokens = hs[-1]
                    #print('second_visual_tokens', second_visual_tokens.shape)
                    output_tokens.append(second_visual_tokens)

                ## video transformer
                if self.video_bb == 'timesformer':
                    hs = self.video_transformer(input_proj_video, video_mask, self.video_query_embed.weight, video_pos_embed, self.return_interm)
                    visual_tokens = hs[-1]
                    output_tokens.append(visual_tokens)

                elif self.video_bb == 'VideoMAEv2':
                    output_tokens.append(video_embeddings.unsqueeze(1).repeat(1, 1000, 1)) # [1, 1000, 768]
                
                elif self.video_bb == 'InternVideo':

                    hs = self.video_transformer(input_proj_video, video_mask, self.video_query_embed.weight, video_pos_embed, self.return_interm)
                    visual_tokens = hs[-1]
                    output_tokens.append(visual_tokens)


            ## audio transformer
            if 'audio' in self.modality:
                if self.audio_bb == 'Wav2Vec':
                    hs = self.audio_transformer(input_proj_audio, audio_mask, self.audio_query_embed.weight, audio_pos_embed, self.return_interm)
                    audio_tokens = hs[-1]
                    output_tokens.append(audio_tokens)

                # audio_tokens = input_proj_audio.permute(0, 3, 1, 2).mean(-1)
                # output_tokens.append(audio_tokens)

                if 'whisper' in self.second_audio_bb:
                    hs = self.second_audio_transformer(second_input_proj_audio, second_audio_mask, self.second_audio_query_embed.weight, second_audio_pos_embed, self.return_interm)
                    second_audio_tokens = hs[-1]
                    #print('second_audio_tokens', second_audio_tokens.shape)
                    output_tokens.append(second_audio_tokens)

            ## text transformer
            if 'text' in self.modality:

                if 'bert' in self.text_bb:
                    hs = self.text_transformer(input_proj_text, text_mask, self.text_query_embed.weight, text_pos_embed, self.return_interm)
                    text_tokens = hs[-1]
                    #print('text_tokens', text_tokens.shape)
                    output_tokens.append(text_tokens)

                if 'clip' in self.second_text_bb:
                    hs = self.second_text_transformer(second_input_proj_text, second_text_mask, self.second_text_query_embed.weight, secondd_text_pos_embed, self.return_interm)
                    second_text_tokens = hs[-1]
                    output_tokens.append(second_text_tokens)

                if self.third_text_bb == 'llama':
                    hs = self.third_text_transformer(third_input_proj_text, third_text_mask, self.third_text_query_embed.weight, third_text_pos_embed, self.return_interm)
                    third_text_tokens = hs[-1]
                    output_tokens.append(third_text_tokens)


            if self.index_trial == 1:
                ind = samples['ind'].cuda() # [b, 1] -> [b]
                # print('index_embed:', self.index_embedding.shape)
                index_embedding = self.index_embedding[ind]
                # print('index_embedding:', index_embedding.shape)  #torch.Size([1, 64])
                output_tokens.append(index_embedding.unsqueeze(1).repeat(1, 1000, 1))


                # text_token = input_proj_text.permute(0, 3, 1, 2).mean(-1)
                # output_tokens.append(text_token)

            if self.combine_modality == True:
                
                combined_input_proj = torch.cat((input_proj_src.flatten(2), input_proj_audio.flatten(2), input_proj_text.flatten(2)), dim=-1)
                combined_mask = torch.cat((mask.flatten(1), audio_mask.flatten(1), text_mask.flatten(1)), dim=-1)
                combined_pos_embed = torch.cat((pos_embed.flatten(2), audio_pos_embed.flatten(2), text_pos_embed.flatten(2)), dim=-1)

                hs = self.combined_transformer(combined_input_proj, combined_mask, self.combined_query_embed.weight, combined_pos_embed, self.return_interm)
                combined_tokens = hs[-1]
                output_tokens.append(combined_tokens)

                combined_second_input_proj = torch.cat((second_visual_input.flatten(2), second_input_proj_audio.flatten(2), second_input_proj_text.flatten(2)), dim=-1)
                combined_second_mask = torch.cat((second_visual_mask.flatten(1), second_audio_mask.flatten(1), second_text_mask.flatten(1)), dim=-1)
                combined_second_pos_embed = torch.cat((second_pos_embed.flatten(2), second_audio_pos_embed.flatten(2), secondd_text_pos_embed.flatten(2)), dim=-1)

                hs = self.second_combined_transformer(combined_second_input_proj, combined_second_mask, self.second_combined_query_embed.weight, combined_second_pos_embed, self.return_interm)
                second_combined_tokens = hs[-1]
                output_tokens.append(second_combined_tokens)


            if 'transformer' in self.readout_mapping:

                # print('output tokens', output_tokens[0].shape) #([1, 1000, 768])
                readout_proj_src = torch.stack(output_tokens).permute(1,3,0,2).flatten(2)
                #print(f'readout_proj_src: {readout_proj_src.shape}')
                b, hidden_dim, num_tokens = readout_proj_src.shape
                #readout_input_proj = torch.cat((input_proj_src.flatten(2), input_proj_audio.flatten(2), input_proj_text.flatten(2)), dim=-1)
                readout_mask = third_text_mask = torch.zeros(b, num_tokens, 1).cuda()
                readout_pos_embed = self.readout_pos_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
                readout_pos_embed = readout_pos_embed.permute(0, 2, 1, 3)

                hs = self.readout_transformer(readout_proj_src, readout_mask, self.readout_query_embed.weight, readout_pos_embed, self.return_interm)
                combined_tokens = hs[-1]
                output_tokens.append(combined_tokens)

            
            output_tokens = torch.cat(output_tokens, dim=2)


            #output_tokens = torch.cat((visual_tokens, text_tokens), dim=2)
            l2_reg = torch.tensor(0.).cuda()

            fmri_pred = {}
            if self.readout_res == 'parcels':

                # TODO make this based on which subject
                if self.sub == 0:
                    fmri_pred_ = self.lin_embed_1(output_tokens[:,:,:])
                    fmri_pred["sub_1"] = torch.diagonal(fmri_pred_, dim1=-2, dim2=-1)

                    fmri_pred_ = self.lin_embed_2(output_tokens[:,:,:])
                    fmri_pred["sub_2"] = torch.diagonal(fmri_pred_, dim1=-2, dim2=-1)

                    fmri_pred_ = self.lin_embed_3(output_tokens[:,:,:])
                    fmri_pred["sub_3"] = torch.diagonal(fmri_pred_, dim1=-2, dim2=-1)

                    fmri_pred_ = self.lin_embed_5(output_tokens[:,:,:])
                    fmri_pred["sub_5"] = torch.diagonal(fmri_pred_, dim1=-2, dim2=-1)

                #fmri_pred = torch.cat((fmri_pred[1], fmri_pred[2], fmri_pred[3], fmri_pred[5]), dim=1)

                else:
                    fmri_pred_ = self.lin_embed(output_tokens[:,:,:])
                    fmri_pred[f"sub_{self.sub}"] = torch.diagonal(fmri_pred_, dim1=-2, dim2=-1)


            elif self.readout_res == "voxels":
                # print("output tokens:", output_tokens.shape)
                fmri_pred_ = self.lin_embed(output_tokens)
                # print("fmri_pred after lin embed:", fmri_pred.shape)
                # print("parcel mask:", self.parcel_mask.shape)
                fmri_pred_ = fmri_pred_ * self.parcel_mask
                # print("fmri_pred after parcel mask:", fmri_pred.shape)
                fmri_pred_ = torch.sum(fmri_pred_, dim=-2)
                # print("fmri_pred after sum:", fmri_pred.shape)

                fmri_pred[f"sub_{self.sub}"] = fmri_pred_

                for param in self.lin_embed.parameters():
                    l2_reg += torch.norm(param)

            # elif self.readout_res == 'all_parcels':
            #     fmri_pred = self.lin_embed(output_tokens[:,0,:])
            #     # fmri_pred = self.relu(fmri_pred)
            #     # fmri_pred = self.lin_embed_2(fmri_pred)

            #     for param in self.lin_embed.parameters():
            #         l2_reg += torch.norm(param)

            # elif self.readout_res == 'hemis':
            #     lh_fmri_pred = self.lh_embed(output_tokens[:,0,:])
            #     rh_fmri_pred = self.rh_embed(output_tokens[:,1,:])
            #     fmri_pred = torch.cat((lh_fmri_pred, rh_fmri_pred), dim=1)

            #     for param in self.lh_embed.parameters():
            #         l2_reg += torch.norm(param)

            #     for param in self.rh_embed.parameters():
            #         l2_reg += torch.norm(param)  

            # elif self.readout_res == 'hemis':
            #     lh_f_pred = self.lh_embed(output_tokens[:,0,:])
            #     rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            # else:
            #     lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
            #     lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

            #     rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
            #     rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            out = {'fmri_pred': fmri_pred, 'output_tokens': output_tokens, 'l2_reg': l2_reg} #, 'output_tokens': output_tokens}

        # elif self.encoder_arch == 'custom_transformer':

        #     hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
        #     output_tokens = hs[-1]

        #     if self.readout_res == 'voxels':

        #         lh_f_pred = self.lh_embed(output_tokens[:,0:self.lh_vs,:])
        #         rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])

        #         lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)
        #         rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)

        #     elif self.readout_res == 'hemis':
        #         lh_f_pred = self.lh_embed(output_tokens[:,0,:])
        #         rh_f_pred = self.rh_embed(output_tokens[:,1,:])

        #     else:
        #         lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
        #         lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

        #         rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
        #         rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

        #     out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        # elif self.encoder_arch == 'spatial_feature':

        #     if self.downsize:
        #         input_proj_src = self.input_proj(input_proj_src)
            
        #     if self.readout_res == 'rois_all':
        #         # only for rois_all
        #         input_proj_src = input_proj_src.flatten(2)
        #         spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
        #         spatial_map = F.softmax(spatial_map, dim=0)
        #         output_tokens = torch.matmul(input_proj_src, spatial_map)
        #         output_tokens = torch.movedim(output_tokens, 1, 2)

        #         lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
        #         lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

        #         rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
        #         rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

        #     elif self.readout_res == 'voxels':
        #         input_proj_src = input_proj_src.flatten(2)
        #         spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
        #         spatial_map = F.softmax(spatial_map, dim=0)
        #         output_tokens = torch.matmul(input_proj_src, spatial_map)
        #         output_tokens = torch.movedim(output_tokens, 1, 2)

        #         lh_f_pred = self.lh_embed(output_tokens[:,:self.lh_vs,:])
        #         lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)

        #         rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])
        #         rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)


        #     out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        # elif self.encoder_arch == 'linear':
        #     #if 'dino' in self.backbone_arch:
        #     input_proj_src = self.input_proj(input_proj_src)
        #     output_tokens = input_proj_src.flatten(1)
        #     lh_f_pred = self.lh_embed(output_tokens)
        #     rh_f_pred = self.rh_embed(output_tokens)

        #     l2_reg = torch.tensor(0.).cuda()
        #     for param in self.lh_embed.parameters():
        #         l2_reg += torch.norm(param)

        #     for param in self.rh_embed.parameters():
        #         l2_reg += torch.norm(param)  

        #     out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens, 'l2_reg': l2_reg}
        

        return out
