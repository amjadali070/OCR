import glob
import os
import shutil
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv
import cv2
from pdf2image import convert_from_path
from pytesseract import pytesseract
from scipy.ndimage import filters, gaussian_filter
from skimage import color
from skimage import filters
import matplotlib.pyplot as plt
import pytesseract as pt

import spacy
from collections import Counter
import io
from torchtext.vocab import vocab
import copy
from typing import Optional, Any, Union, Callable

import torch
import math
import time
import torch.nn as nn  # wrap the function prarmeters and layers
from torch import Tensor
import torch.nn.functional as F  # help you build neural network model
from torch.nn import Module
from torch.nn import MultiheadAttention  # focus main words of sentrance
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

from flask import Flask, jsonify, request

global imgpath

app = Flask(__name__)


def Scanimage(input_img):
    # 1st
    with Image.open(r"" + input_img) as im:
        new_im = Image.new('RGB', im.size)

        width, height = im.size
        if width == 1280 and height == 720:
            print("Your Picture is Already of standard Size, i.e:")
            print("Width is: ", width)
            print("Height is: ", height)
        else:
            print("The size of Original Image: ")
            print("Width is: ", width)
            print("Height is: ", height)

            new_width = 1280
            new_height = 720
            new_im = im.resize((new_width, new_height))
            print("\nThe Size of Pre-processed Image: ")
            print("Width is: ", new_width)
            print("Height is: ", new_height)

        resonse = resize(new_im)
        return resonse


def resize(new_im):
    imgpath = r"images/process_images/resized.png"
    newimage = new_im.save(imgpath)

    imgpath = imgpath
    Input_Image = cv.imread(imgpath, 0)
    Histogram_Grayscale = Histogram_Computation(Input_Image)
    Plot_Histogram(Histogram_Grayscale, "Histogram_Grayscale")
    response = conversions(imgpath)
    return response


def conversions(imgpath):
    img = cv2.imread(imgpath)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    aspect_ratio = float(img.shape[0]) / float(img.shape[1])

    # Calculate the new dimensions, keeping the aspect ratio the same
    new_height = 600
    new_width = int(new_height / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # =============================================================================

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    aspect_ratio = float(equalized.shape[0]) / float(equalized.shape[1])

    # Calculate the new dimensions, keeping the aspect ratio the same
    new_height = 400
    new_width = int(new_height / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(equalized, (new_width, new_height))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Save the output image
    new_path = "images/process_images/Eq.png"
    cv2.imwrite(new_path, equalized)
    cv2.waitKey(0)

    img = cv2.imread(new_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=5)
    final = clahe.apply(gray)

    newimg_path = "images/process_images/CLAHE.png"
    # cv2.imshow('Adaptive Equalized', final)
    cv2.imwrite(newimg_path, final)
    cv2.waitKey(0)
    response = plots(newimg_path)
    return response


def Histogram_Computation(Image):
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    Histogram = np.zeros([256], np.int32)

    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            Histogram[Image[x, y]] += 1

    return Histogram


def Plot_Histogram(Histogram, image_name):
    plt.figure()
    plt.title("GrayScale Histogram")
    plt.xlabel("Intensity Level")
    plt.ylabel("Intensity Frequency")
    plt.xlim([0, 285])
    plt.plot(Histogram)
    plt.savefig("images/process_images/" + image_name + ".jpg")


def plots(newimg_path):
    Input_Image = cv.imread(newimg_path, 0)
    Histogram_Grayscale = Histogram_Computation(Input_Image)
    Plot_Histogram(Histogram_Grayscale, "Final_Histogram")

    img = cv2.imread(newimg_path)
    image = color.rgb2gray(img)
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold

    # Plot the original and the binary image
    plt.figure()

    if threshold < 0.6 and threshold > 0.4:
        res = "Your Image is not Blur. "
        # print(res)

    else:
        res = "Your Image is Blur."
        # print(res)
        img = cv2.imread(newimg_path, 0)

        guss_blur = cv2.GaussianBlur(img, (7, 7), 2)
        sharp = cv2.addWeighted(img, 4.5, guss_blur, -3.5, 0)

        final_path = "images/process_images/sharp.png"
        cv2.imwrite(final_path, sharp)
        cv2.waitKey(0)

    return extract_Text(newimg_path, final_path, res)


def extract_Text(newimg_path, final_path, res):
    responce = ""
    if res == "Your Image is not Blur. ":

        final_path = newimg_path
        pt.pytesseract.tesseract_cmd = (r'C:/Program Files/Tesseract-OCR/tesseract.exe')
        img = Image.open(final_path)
        img1 = np.array(Image.open(final_path))
        data = pt.image_to_string(img1, lang='eng', config='--psm 6')

        print("-------------------------")
        print("Extracted Text from Image")
        print("-------------------------\n")
        if data == "":
            print("No Text Found")
            responce = "No Text Found"
            return responce
        else:
            # print(data)
            responce = data
            return responce
        # print(res)
    else:
        final_path = final_path
        pt.pytesseract.tesseract_cmd = (r'C:/Program Files/Tesseract-OCR/tesseract.exe')
        img = Image.open(final_path)
        img1 = np.array(Image.open(final_path))
        data = pt.image_to_string(img1, lang='eng', config='--psm 6')

        print("-------------------------")
        print("Extracted Text from Image")
        print("-------------------------\n")
        if data == "":
            print("No Text Found")
            responce = "No Text Found"
            return responce
        else:
            # print(data)
            responce = data
            return responce
        # print(res)


def extractImgFromPDF(pdf):
    pdfs = glob.glob(r"" + pdf)
    count = 0
    global img_path
    img_path = []
    for pdf_path in pdfs:
        with TemporaryDirectory() as tempdir:
            pages = convert_from_path(pdf_path, 500)
            for page_enumeration, page in enumerate(pages, start=1):
                filename = f"{'images/PDF_2_image'}/page_{page_enumeration:03}.jpg"
                count += 1
                img_path.append(filename)

                page.save(filename, "JPEG")
                print("Saved Successfully at: ", filename)

            print("Your PDF has", count, "Images.")
            print(img_path)

    res = resizingImgPDF(img_path)
    return res


def resizingImgPDF(img_path):
    total = len(img_path)
    img_name = []

    for i in img_path:
        with Image.open(str(i)) as im:
            new_im = Image.new('RGB', im.size)

            width, height = im.size

            new_width = 1280
            new_height = 720
            new_im = im.resize((new_width, new_height))

            directory = os.path.dirname(i)
            filename = os.path.basename(i)

            names = "image" + str(img_path.index(i))
            img_name.append(names)

            for n in img_name:
                new = n + "_" + filename
                new_path = os.path.join(directory, new)
                shutil.copy(i, new_path)
            print("Image saved as:", new_path)

    res = PlotingHistogramtotheCorrespondingImagePDF()
    return res


def PlotingHistogramtotheCorrespondingImagePDF():
    for i in img_path:
        Input_Image = cv.imread(i, 0)
        Histogram_Grayscale = Histogram_ComputationPDF(Input_Image)
        # print(Histogram_Grayscale)
        # for i in range(0, len(Histogram_Grayscale)):
        #     print("Histogram[", i, "]: ", Histogram_Grayscale[i])
        Plot_HistogramPDF(Histogram_Grayscale, "Histogram_Grayscale")

    res = EquilizatingTheHistogramPDF()
    return res


def EquilizatingTheHistogramPDF():
    for i in img_path:
        img = cv2.imread(i)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        equalized = cv2.equalizeHist(gray)

        new_path = "images/PDFproc/Eq.png"

        cv2.waitKey(0)

    res = HistogramAdaptiveEquilizationPDF()
    return res


def HistogramAdaptiveEquilizationPDF():
    for i in img_path:
        img = cv2.imread(i)
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)

        # Save the output image
        cv2.imwrite("images/PDFproc/res.png", equalized)

        clahe = cv2.createCLAHE(clipLimit=5)
        final = clahe.apply(gray)
        newimg_path = "images/PDFproc/sam7_CLAHE.png"
        cv2.imwrite(newimg_path, final)

    res = HistogramAdaptiveEquilizationMainPDF()
    return res


def HistogramAdaptiveEquilizationMainPDF():
    for i in img_path:
        Input_Image = cv.imread(i, 0)
        Histogram_Grayscale = Histogram_ComputationPDF(Input_Image)

        Plot_HistogramPDF(Histogram_Grayscale, "Final_Histogram")

    res = BlurDetectionPDF()
    return res


def BlurDetectionPDF():
    for i in img_path:
        img = cv2.imread(i)
        image = color.rgb2gray(img)

        # threshold = filters.threshold_otsu(image)
        threshold = gaussian_filter(image, 10)

        binary_image = image > threshold

        # Plot the original and the binary image
        plt.figure()
        # plt.subplot(1, 2, 1)
        print("Original Image ", "    |     ", " Binary Image")
        # plt.imshow(image, cmap="gray")
        # plt.subplot(1, 2, 2)
        # plt.imshow(binary_image, cmap="gray")
        # plt.show()

        # if threshold < 0.7 and threshold > 0.3:
        #     print("Your Image is not Blur. ")
        # else:
        #     print("Your Image is Blur.")

    res = HandlingBluredImagePDF()
    return res


def HandlingBluredImagePDF():
    for i in img_path:
        img = cv2.imread(i, 0)

        guss_blur = cv2.GaussianBlur(img, (7, 7), 2)
        sharp = cv2.addWeighted(img, 4.5, guss_blur, -3.5, 0)
        final_path = "images/PDFproc/sharp.png"
        cv2.imwrite(final_path, sharp)
        cv2.waitKey(0)

    res = extract_text_from_filesPDF(img_path)
    return res


def extract_text_from_filesPDF(img_path):
    extractList = []
    for i in img_path:
        try:
            file_name = os.path.splitext(os.path.basename(i))[0]

            if i.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                pytesseract.tesseract_cmd = ('C:/Program Files/Tesseract-OCR/tesseract.exe')
                image = Image.open(i)
                text = pytesseract.image_to_string(image)
                extractList.append(text)
                print("Text extracted from", i, "and saved to")
            else:
                print("Unsupported file format:")
                return "Unsupported file format:"

        except FileNotFoundError:
            print("File not found:")
            return "File not found:"

    return extractList


def Histogram_ComputationPDF(Image):
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    Histogram = np.zeros([256], np.int32)

    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            Histogram[Image[x, y]] += 1

    return Histogram


def Plot_HistogramPDF(Histogram, name):
    plt.figure()
    plt.title("GrayScale Histogram")
    plt.xlabel("Intensity Level")
    plt.ylabel("Intensity Frequency")
    plt.xlim([0, 285])
    plt.plot(Histogram)
    plt.savefig("images/PDFproc/" + name + ".jpg")


# ------------- Code for Translation --------------
nlp = spacy.blank('ur')


def urdu_tokens(inp):
    # print("The inp to urdu tokens is :",inp)
    doc = nlp(inp)
    lst = []
    lst.clear()
    for word in doc:
        word = str(word)
        lst.append(word)

    return lst


nlp = spacy.blank('ur')


# generating token for input urdu text
def urdu_sen_tokens(inp):
    # print("The inp to urdu tokens is :",inp)
    doc = nlp(inp)
    doc = str(doc)
    doc = doc.replace("\n", "")
    return [doc]


nlp = spacy.blank('en')


# generating token for input english text
def eng_sen_tokens(inp):
    # print("The inp to eng tokens is :",inp)
    doc = nlp(inp)
    doc = str(doc)
    # print("doc type is :",type(doc))

    # print("ret type is ",type([doc]))
    doc = doc.replace("\n", "")
    return [doc]


class Transformer(Module):

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # encoding forward pass
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)  # decoding forward pass
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        if isinstance(first_layer, torch.nn.TransformerEncoderLayer):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor):
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not torch.overrides.has_torch_function(tensor_args):
                        if output.is_cuda or 'cpu' in str(output.device):
                            convert_to_nested = True
                            output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not())

        for mod in self.layers:
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)

        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        if (src.dim() == 3 and not self.norm_first and not self.training and
                self.self_attn.batch_first and
                self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
                self.norm1.eps == self.norm2.eps and
                ((src_mask is None and src_key_padding_mask is None)
                if src.is_nested
                else (src_mask is None or src_key_padding_mask is None))):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and

                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,
                )
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)

        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


torch.use_deterministic_algorithms(True)

fileEnglish = open('English.txt', mode='rt', encoding='utf-8')
englishDataset = fileEnglish.read()
print(englishDataset[0:500000])
file = open("English.txt", "r", encoding='utf-8')
x = 0
for line in file:

    if line != "\n":
        x += 1
    # print(line)
file.close()

torch.use_deterministic_algorithms(True)
fileUrdu = open('Urdu.txt', mode='rt', encoding='utf-8')
urduDataset = fileUrdu.read()
print(urduDataset[0:500000])

file = open("Urdu.txt", "r", encoding='utf-8')
x = 0
for line in file:

    if line != "\n":
        x += 1
    # print((x))
file.close()

train_size_en = int(0.70 * x)
test_size_en = int(0.15 * x)
val_size_en = int(0.15 * x)
# division on the basis of lines


train_dataset = englishDataset[0:train_size_en]
test_dataset = englishDataset[train_size_en + 1:train_size_en + test_size_en]
val_dataset = englishDataset[train_size_en + test_size_en + 1:x]

train_size_urdu = int(0.70 * x)
test_size_urdu = int(0.15 * x)
val_size_urdu = int(0.15 * x)

train_dataset2 = urduDataset[0:train_size_urdu]
test_dataset2 = urduDataset[train_size_urdu + 1:train_size_urdu + test_size_urdu]
val_dataset2 = urduDataset[train_size_urdu + test_size_urdu + 1:x]

f = open("english_train.txt", "w", encoding="utf-8")

f.write(train_dataset)

f = open("english_test.txt", "w", encoding="utf-8")
f.write(test_dataset)

f = open("english_val.txt", "w", encoding="utf-8")
f.write(val_dataset)

f = open("urdu_train.txt", "w", encoding="utf-8")

f.write(train_dataset2)

f = open("urdu_test.txt", "w", encoding="utf-8")
f.write(test_dataset2)

f = open("urdu_val.txt", "w", encoding="utf-8")
f.write(val_dataset2)

de_tokenizer = urdu_sen_tokens
en_tokenizer = eng_sen_tokens


def build_vocab(filepath, tokenizer):
    counter = Counter()  # counter builds dictionary of word with its frequencies
    with io.open(filepath, encoding="utf8", errors='ignore') as f:
        for string_ in f:
            # print(tokenizer(string_))
            counter.update(tokenizer(string_))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


main_filepaths = ["English.txt", "Urdu.txt"]
train_filepaths = ["english_train.txt", "urdu_train.txt"]
val_filepaths = ["english_val.txt", "urdu_val.txt"]
test_filepaths = ["english_test.txt", "urdu_test.txt"]
en_vocab = build_vocab(train_filepaths[0], en_tokenizer)
de_vocab = build_vocab(train_filepaths[1], de_tokenizer)


def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[1], encoding="utf8", errors='ignore'))
    raw_en_iter = iter(io.open(filepaths[0], encoding="utf8", errors='ignore'))
    data = []
    for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de.rstrip("\n"))],
                                  dtype=torch.long)

        en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en.rstrip("\n"))],
                                  dtype=torch.long)
        data.append((de_tensor_, en_tensor_))

    return data


de_vocab.set_default_index(0)
en_vocab.set_default_index(0)
train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def generate_batch(data_batch):
    de_batch, en_batch = [], []
    for (de_item, en_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))

        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True,
                        collate_fn=generate_batch)  # dividing the data in batches with the batch size specified
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))

        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)

        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)

        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


SRC_VOCAB_SIZE = len(de_vocab)
TGT_VOCAB_SIZE = len(en_vocab)

EMB_SIZE = 512

NHEAD = 8

FFN_HID_DIM = 512

BATCH_SIZE = 1024

NUM_ENCODER_LAYERS = 3

NUM_DECODER_LAYERS = 3

NUM_EPOCHS = 8

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(valid_iter)):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)


for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_iter, optimizer)
    end_time = time.time()
    val_loss = evaluate(transformer, valid_iter)

    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
           f"Epoch time = {(end_time - start_time):.3f}s"))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab.get_stoi()[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join([tgt_vocab.get_itos()[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")


file1 = open('english_train.txt', 'r', encoding='utf8')
Lines = file1.readlines()

file2 = open('trans.txt', 'w', encoding='utf8')
count = 0

for line in Lines:
    count += 1
    print(line.strip())
    var = translate(transformer, line.strip(), en_vocab, de_vocab, en_tokenizer)
    print(var)
    file2.writelines(var)
    if count == 150:
        break


def Translation(file1_path, file2_path, target_sentence):
    with open(file1_path, 'r', encoding='utf-8') as file1:
        sentences_file1 = file1.readlines()
    with open(file2_path, 'r', encoding='utf-8') as file2:
        sentences_file2 = file2.readlines()
    matching_sentences = []
    for line_num, sentence in enumerate(sentences_file1, 1):
        if target_sentence in sentence:
            matching_sentences.append((line_num, sentence.strip()))
    if matching_sentences:
        print(f"English Translation")
        for line_num, sentence in matching_sentences:
            print(f"Line {line_num}: {sentence}")

            if line_num <= len(sentences_file2):
                print(f"Urdu Translation:")
                print(f"Line {line_num}: {sentences_file2[line_num - 1].strip()}")
                res = sentences_file2[line_num - 1].strip()
                return res
            else:
                print(f"No sentence form {line_num}")
                res = f"No sentence form {line_num}"
                return res
    else:
        print("No Sentence or not in Dataset.")
        res = "No Sentence or not in Dataset."
        return res


# ----------------------------------------


@app.route('/process_image', methods=['POST'])
def process_image():
    # print(request.files.)
    if 'image' not in request.files:
        return "No image found"
    image_file = request.files['image']
    file_name = image_file.filename
    folder_path = "images/original_images/"  # Specify the folder where you want to save the image
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

    image_file.save(os.path.join(folder_path, file_name))

    response = Scanimage("images/original_images/" + file_name)

    print(response)

    if (response != ""):
        result = {
            'status': 'success',
            'text': response
            # Add any other result data you want to return
        }
    else:
        result = {
            'status': 'Error',
            'text': 'Something went wrong'
            # Add any other result data you want to return
        }

    return jsonify(result), 200


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # print(request.files.)
    if 'pdf' not in request.files:
        return "No pdf found"
    pdf_file = request.files['pdf']

    pdf_file = request.files['pdf']
    file_path = 'images/PDF_2_image/' + pdf_file.filename  # Specify the directory path where you want to save the file

    pdf_file.save(file_path)

    response = extractImgFromPDF(file_path)

    if (response != ""):
        result = {
            'status': 'success',
            'text': response
            # Add any other result data you want to return
        }
    else:
        result = {
            'status': 'Error',
            'text': 'Something went wrong'
            # Add any other result data you want to return
        }

    return jsonify(result), 200


@app.route('/eng_to_urdu_translate', methods=['POST'])
def eng_to_urdu_translate():
    # Get the input sentence from the request
    eng_text = request.json['engText']

    file1_path = r'D:\Personal\OCR\3rd milestone\english_train.txt'
    file2_path = r'D:\Personal\OCR\3rd milestone\urdu_train.txt'
    # Perform translation using your model
    translation = Translation(file1_path, file2_path, eng_text)

    # Return the translation as the API response
    response = {'translation': translation}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host="192.168.0.113", port=5000)