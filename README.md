# Transformer_pytorch


## Project Overview 📊
Use pytorch to build Transformer model structure
<div align="center">
  <img src="images/Transformer.png" width="500" />
</div>

## Step1: Build the Attention layer 🛠️
<div align="center">
  <img src="images/Attention2.png" width="100" /><img src="images/Attention.png" width="100" />
</div>

### Multi-head attention: 
The input at each position can simultaneously focus on all other positions in the sequence. This means that each position can receive information from all other positions in the input sequence.

### Masked Multi-head attention: 
In the Decoder, since the order from left to right needs to be followed when generating the output sequence, at each time step, the model can only focus on the position before the current position, rather than the entire input sequence. This means that during the process of generating the sequence, the model cannot see the information behind the current position to avoid information leakage.
In order to achieve this, the Masked Attention in the Decoder will mask out the information after the current position when calculating the Attention weight, usually using a mask matrix.

## Step2: Build the Transformer Block 🛠️
<div align="center">
  <img src="images/TransformerBlock.png" width="150" />
</div> 

### Layer Normalization 
Layer Normalization: Calculate the mean and variance of the output of each layer and use it to normalize the output of the layer. It does not rely on the batch size or batch statistics, but only focuses on the output of each layer.


## Step3: Build the Encoder 🛠️
<div align="center">
  <img src="images/Encoder.png" width="150" />
</div>

Perform Input Embedding and Positional Encoding on the input. In addition, set N Transformer Blocks.


## Step4: Build the Decoder Block 🛠️
<div align="center">
  <img src="images/DecoderBlock.png" width="150" />
</div>

The first is to write the Attention structure with Mask. Then, because the above structure is the same as the Transformer Block in Encoder, we call it directly.

## Step5: Build the Decoder 🛠️
<div align="center">
  <img src="images/Decoder.png" width="150" />
</div>

Perform Input Embedding and Positional Encoding on the input. In addition, set N Decoder Blocks.  

Then, Linear and softmax.

## Step6: Combine the above structures together to form the Transformer model 🛠️

First, a mask is generated for the input sequence using the make_src_mask and make_trg_mask methods.  

The source sequence and its mask are then passed to the encoder, which returns a continuous representation of the source sequence.  

Finally, the target sequence, the output of the encoder, the source sequence mask, and the target sequence mask are passed to the decoder, which generates the final output.  







