import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from utils.utils import (NestedTensor, nested_tensor_from_tensor_list)

from models.backbone import build_backbone
from models.transformer import build_transformer
from models.custom_transformer import build_custom_transformer
from models.position_encoding import build_position_encoding

from transformers import BertModel, GPT2Model
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa



#from transformers import ASTModel, ASTProcessor

# # Load audio
# y, sr = librosa.load('path_to_audio.wav', sr=16000)

# # Initialize AST model and processor
# processor = ASTProcessor.from_pretrained("facebook/ast-base")
# model = ASTModel.from_pretrained("facebook/ast-base")

# # Preprocess the audio
# inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)

# # Extract features
# with torch.no_grad():
#     outputs = model(**inputs)

# feature_tokens = outputs.last_hidden_state
# print(feature_tokens.shape)



# from transformers import BertTokenizer, BertModel

# # Load pre-trained BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertModel.from_pretrained("bert-base-uncased")

# # Input text
# text = "This is an example sentence."

# # Tokenize the text
# inputs = tokenizer(text, return_tensors="pt")

# # Extract token embeddings
# with torch.no_grad():
#     outputs = model(**inputs)

# # Extract token-level features (token embeddings)
# token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
# print(token_embeddings.shape)


# from transformers import GPT2Tokenizer, GPT2Model
# import torch

# # Load GPT-2 tokenizer and model
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2Model.from_pretrained("gpt2")

# # Input text
# text = "This is an example sentence."

# # Tokenize the text
# inputs = tokenizer(text, return_tensors="pt")

# # Extract token embeddings
# with torch.no_grad():
#     outputs = model(**inputs)

# # Extract token-level features (token embeddings)
# token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
# print(token_embeddings.shape)


class brain_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.lr_backbone = args.lr_backbone

        self.backbone_arch = args.backbone_arch
        self.return_interm = args.return_interm
        self.encoder_arch = args.encoder_arch
        self.modality = args.modality

        ### backbone_arch for feature exraction
        self.backbone_model = build_backbone(args)

        # number of brain areas
        self.num_queries = args.num_queries

        #TODO hard  coding the map size for now but fix it
        self.map_size = 31

        # self.audio_processor = ASTProcessor.from_pretrained("facebook/ast-base")
        # self.audio_model = ASTModel.from_pretrained("facebook/ast-base")

        ### Brain encoding model
        if 'transformer' in args.encoder_arch:
            if args.encoder_arch == 'transformer':

                self.hidden_dim = 768 # self.transformer.d_model
                self.linear_feature_dim  = self.hidden_dim

                if 'visual' in args.modality:
                    self.transformer = build_transformer(args)
                    self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                if 'audio' in args.modality:
                    self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                    for param in self.audio_model.parameters():
                        param.requires_grad = False
                    self.audio_transformer = build_transformer(args)
                    self.audio_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                if 'text' in args.modality:
                    self.lang_model = BertModel.from_pretrained("bert-base-uncased")
                    #self.lang_model = GPT2Model.from_pretrained("gpt2")
                    for param in self.lang_model.parameters():
                        param.requires_grad = False
                    self.text_transformer = build_transformer(args)
                    self.text_query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

                # self.audio_transformer = build_transformer(args)
                # self.text_transformer = build_transformer(args)

            elif self.encoder_arch == 'custom_transformer':
                self.transformer = build_custom_transformer(args)

            if ('resnet' in self.backbone_arch):
                self.input_proj = nn.Conv2d(self.backbone_model.num_channels, self.hidden_dim, kernel_size=1)
        
        elif self.encoder_arch == 'spatial_feature':

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
            self.time_embed = nn.Embedding(20, self.hidden_dim)
            
        
        # self.position_embedding = build_position_encoding(args.position_embedding, args.hidden_dim // 2)
        # b = 1 # brain_data.shape[0]
        # x = torch.ones(b, self.hidden_dim, 20, 1).cuda() # TODO # frames
        # x = nested_tensor_from_tensor_list(tuple(x))
        # time_embed = self.position_embedding(x)
        # time_embed = time_embed.squeeze().flatten(1).permute(1, 0)
        #self.time_embed = time_embed.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) 
        
        # print('time_embed:', self.time_embed.shape)

        ## audio input
        if 'audio' in args.modality:
            #self.audio_proj = nn.Conv2d(40, self.hidden_dim, kernel_size=1)
            self.audio_time_embed = nn.Embedding(1117, self.hidden_dim)
            

        ## text input
        if 'text' in args.modality:
            self.text_time_embed = nn.Embedding(128, self.hidden_dim)
            

        self.readout_res = args.readout_res

        if self.readout_res == 'voxels' or self.readout_res == 'all_parcels':
            self.vs = 1000
            self.lin_embed = nn.Sequential(
                nn.Linear(len(args.modality.split(' '))*self.linear_feature_dim, self.vs),
            )
            # self.relu = nn.ReLU()
            # self.lin_embed_2 = nn.Sequential(
            #     nn.Linear(self.vs//2, self.vs),
            # )
        elif self.readout_res == 'hemis':
            self.vs = 500
            self.lh_embed = nn.Sequential(
                nn.Linear(len(args.modality.split(' '))*self.linear_feature_dim, self.vs),
            )
                        
            self.rh_embed = nn.Sequential(
                nn.Linear(len(args.modality.split(' '))*self.linear_feature_dim, self.vs),
            )
        
        # self.rh_embed = nn.Sequential(
        #     nn.Linear(self.linear_feature_dim, 1000),
        # )
            

    def forward(self, samples: NestedTensor):

        #### visual input
        if 'visual' in self.modality:
            frames = samples['visual'] #[:,8].cuda()
            
            b, f, c, h, w = frames.shape
            frames = frames.view(b*f, c, h, w).cuda()

            if isinstance(frames, (list, torch.Tensor)):
                frames = nested_tensor_from_tensor_list(frames)

            # if self.backbone_arch:
            if self.lr_backbone == 0:
                with torch.no_grad():
                    features, pos = self.backbone_model(frames)
            else:
                features, pos = self.backbone_model(frames)

            input_proj_src, mask = features[-1].decompose()
            assert mask is not None
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

            time_embed = self.time_embed.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
            time_embed = time_embed.repeat(b, 1, 1, h, w)
            pos_embed = pos_embed + time_embed
            pos_embed = pos_embed.permute(0, 2, 1, 3, 4)

        #### audio inputw
        if 'audio' in self.modality:
            audio = samples['audio'].cuda() 
            # sr = samples['sr'].cuda()[0]
            b, _= audio.shape

            with torch.no_grad():
                input_audio = self.audio_model(audio).last_hidden_state

            b, seq, _ = input_audio.shape

            # input_proj_audio = self.audio_proj(input_audio.unsqueeze(-1))
            input_proj_audio= input_audio.unsqueeze(-1).permute(0, 2, 1, 3)
            audio_time_embed = self.audio_time_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
            audio_pos_embed = audio_time_embed.permute(0, 2, 1, 3)
            audio_mask = torch.zeros(b, seq, 1).cuda()

            #print('input_proj_audio:', input_proj_audio.shape)

            # input_proj_audio: torch.Size([1, 768, 482, 1])
            # audio_pos_embed: torch.Size([1, 768, 482, 1])

            # Preprocess the audio
            # audio = self.audio_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

            # with torch.no_grad():
            #     outputs = self.audio_model(**audio)

            # audio_tokens = outputs.last_hidden_state
            # print(audio_tokens.shape)

        #### text input
        if 'text' in self.modality:
            text = samples['text']
            text = {key: value.squeeze(1).cuda() for key, value in text.items()}

            # Extract token embeddings
            with torch.no_grad():
                text_embeddings = self.lang_model(**text)

            text_embeddings = text_embeddings.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
            #text_embeddings = text_embeddings['pooler_output'].unsqueeze(1)

            input_proj_text = text_embeddings.permute(0, 2, 1).unsqueeze(-1)
            text_time_embed = self.text_time_embed.weight.unsqueeze(0).unsqueeze(-1).cuda()
            text_pos_embed = text_time_embed.permute(0, 2, 1, 3)
            text_mask = torch.zeros(text_embeddings.shape[0], 128, 1).cuda()

            #print('input_proj_text:', input_proj_text.shape)

        if self.encoder_arch == 'transformer':
            
        # if backbone is resnet, apply 1x1 conv to project the feature to the transformer dimension
            if 'resnet' in self.backbone_arch:
                input_proj_src = self.input_proj(input_proj_src)

            output_tokens = []

            ## visual transformer
            if 'visual' in self.modality:
                hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
                visual_tokens = hs[-1]
                output_tokens.append(visual_tokens)

            ## audio transformer
            if 'audio' in self.modality:
                hs = self.audio_transformer(input_proj_audio, audio_mask, self.audio_query_embed.weight, audio_pos_embed, self.return_interm)
                audio_tokens = hs[-1]
                output_tokens.append(audio_tokens)

                # audio_tokens = input_proj_audio.permute(0, 3, 1, 2).mean(-1)
                # output_tokens.append(audio_tokens)

            ## text transformer
            if 'text' in self.modality:
                hs = self.text_transformer(input_proj_text, text_mask, self.text_query_embed.weight, text_pos_embed, self.return_interm)
                text_tokens = hs[-1]
                output_tokens.append(text_tokens)

                # text_token = input_proj_text.permute(0, 3, 1, 2).mean(-1)
                # output_tokens.append(text_token)

            output_tokens = torch.cat(output_tokens, dim=2)

            #print('text_tokens:', text_tokens.shape)

            #output_tokens = torch.cat((visual_tokens, text_tokens), dim=2)
            l2_reg = torch.tensor(0.).cuda()

            if self.readout_res == 'voxels':
                fmri_pred = self.lin_embed(output_tokens[:,:,:])
                fmri_pred = torch.diagonal(fmri_pred, dim1=-2, dim2=-1)

                for param in self.lin_embed.parameters():
                    l2_reg += torch.norm(param)

            elif self.readout_res == 'all_parcels':
                fmri_pred = self.lin_embed(output_tokens[:,0,:])
                # fmri_pred = self.relu(fmri_pred)
                # fmri_pred = self.lin_embed_2(fmri_pred)

                for param in self.lin_embed.parameters():
                    l2_reg += torch.norm(param)

            elif self.readout_res == 'hemis':
                lh_fmri_pred = self.lh_embed(output_tokens[:,0,:])
                rh_fmri_pred = self.rh_embed(output_tokens[:,1,:])
                fmri_pred = torch.cat((lh_fmri_pred, rh_fmri_pred), dim=1)

                for param in self.lh_embed.parameters():
                    l2_reg += torch.norm(param)

                for param in self.rh_embed.parameters():
                    l2_reg += torch.norm(param)  

            # elif self.readout_res == 'hemis':
            #     lh_f_pred = self.lh_embed(output_tokens[:,0,:])
            #     rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            # else:
            #     lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
            #     lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

            #     rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
            #     rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            out = {'fmri_pred': fmri_pred, 'output_tokens': output_tokens, 'l2_reg': l2_reg} #, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'custom_transformer':

            hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.return_interm)
            output_tokens = hs[-1]

            if self.readout_res == 'voxels':

                lh_f_pred = self.lh_embed(output_tokens[:,0:self.lh_vs,:])
                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])

                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)

            elif self.readout_res == 'hemis':
                lh_f_pred = self.lh_embed(output_tokens[:,0,:])
                rh_f_pred = self.rh_embed(output_tokens[:,1,:])

            else:
                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'spatial_feature':

            if self.downsize:
                input_proj_src = self.input_proj(input_proj_src)
            
            if self.readout_res == 'rois_all':
                # only for rois_all
                input_proj_src = input_proj_src.flatten(2)
                spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
                spatial_map = F.softmax(spatial_map, dim=0)
                output_tokens = torch.matmul(input_proj_src, spatial_map)
                output_tokens = torch.movedim(output_tokens, 1, 2)

                lh_f_pred = self.lh_embed(output_tokens[:,:output_tokens.shape[1]//2,:])
                lh_f_pred = torch.movedim(lh_f_pred, 1,-1)

                rh_f_pred = self.rh_embed(output_tokens[:,output_tokens.shape[1]//2:,:])
                rh_f_pred = torch.movedim(rh_f_pred, 1,-1)

            elif self.readout_res == 'voxels':
                input_proj_src = input_proj_src.flatten(2)
                spatial_map = torch.transpose(self.spatial_embed.weight, 0, 1)
                spatial_map = F.softmax(spatial_map, dim=0)
                output_tokens = torch.matmul(input_proj_src, spatial_map)
                output_tokens = torch.movedim(output_tokens, 1, 2)

                lh_f_pred = self.lh_embed(output_tokens[:,:self.lh_vs,:])
                lh_f_pred = torch.diagonal(lh_f_pred, dim1=-2, dim2=-1)

                rh_f_pred = self.rh_embed(output_tokens[:,self.lh_vs:,:])
                rh_f_pred = torch.diagonal(rh_f_pred, dim1=-2, dim2=-1)


            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}

        elif self.encoder_arch == 'linear':
            #if 'dino' in self.backbone_arch:
            input_proj_src = self.input_proj(input_proj_src)
            output_tokens = input_proj_src.flatten(1)
            lh_f_pred = self.lh_embed(output_tokens)
            rh_f_pred = self.rh_embed(output_tokens)

            l2_reg = torch.tensor(0.).cuda()
            for param in self.lh_embed.parameters():
                l2_reg += torch.norm(param)

            for param in self.rh_embed.parameters():
                l2_reg += torch.norm(param)  

            out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens, 'l2_reg': l2_reg}
        

        return out
