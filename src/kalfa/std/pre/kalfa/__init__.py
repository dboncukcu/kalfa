from kalfa.registration import pack


lego = pack(__name__)


lego("/pre/kalfa/char_tokenizer", "char_tokenizer:CharTokenizer", state=True, alias="char_tokenizer",
     description="Character level tokenizer fitted on the train text; the vocabulary goes into the record")

lego("/pre/kalfa/cast", "encoders:Cast", alias="cast", description="Cast a column to a numpy dtype")
lego("/pre/kalfa/one_hot", "encoders:OneHot", state=True, alias="one_hot",
     description="One hot columns <field>_<category> of a categorical column; unknown categories give zeros")
lego("/pre/kalfa/label_encoder", "encoders:LabelEncoder", state=True, alias="label_encoder",
     description="Integer codes of a label column, sorted by label; inverted in reports and predictions, class scores "
                 "decode to labels")

lego("/pre/kalfa/resize", "images:Resize", alias="resize", description="Resize an image to size (int or [h, w])")
lego("/pre/kalfa/to_tensor", "images:ToTensor", alias="to_tensor",
     description="Image to a float tensor in [0, 1], channels first")
lego("/pre/kalfa/to_tensor_signed", "images:ToTensorSigned", alias="to_tensor_signed",
     description="Image to a float tensor in [-1, 1], channels first")
lego("/pre/kalfa/normalize", "images:Normalize", alias="normalize",
     description="Normalize an image tensor per channel; mean and std are numbers, lists or the presets imagenet and "
                 "cifar10")
lego("/pre/kalfa/random_crop_flip", "images:RandomCropFlip", alias="random_crop_flip",
     description="Random crop of size after padding and a random horizontal flip")
lego("/pre/kalfa/simclr_aug", "images:SimclrAug", alias="simclr_aug",
     description="SimCLR augmentation: random resized crop to size, horizontal flip, brightness jitter")
lego("/pre/kalfa/two_views", "images:TwoViews", alias="two_views", refs={"transform": "preprocessor"},
     description="Two independent applications of a transform to one image, as a pair")

lego("/pre/kalfa/median_std_scaler", "median_std_scaler:MedianStdScaler", alias="median_std_scaler", grouped=True,
     description="Center a column on its median and scale it by its standard deviation: the center an outlier does "
                 "not move, the scale of a standard scaler")

lego("/pre/kalfa/simple_imputer", "missing:SimpleImputer", alias="simple_imputer",
     description="Fill the missing values of a column with the train mean, median, most frequent value or a constant; "
                 "indicator adds <field>_missing, computed before the fill, as a feature of its own")
lego("/pre/kalfa/fill", "missing:Fill", alias="fill",
     description="Fill the missing values of a column without a fit: a constant (a number, or a name such as missing "
                 "that becomes its own category), or method ffill or bfill along the rows")

lego("/pre/kalfa/abs", "scales:Absolute", alias="abs", description="Absolute value of a column")
lego("/pre/kalfa/log", "scales:Log", alias="log", description="log1p of a column divided by norm, in the given base")
lego("/pre/kalfa/asinh", "scales:Asinh", alias="asinh",
     description="Signed log scale of a heavy tailed column: arcsinh(x / scale), inverted by scale sinh(y); keeps the "
                 "sign, linear near zero, logarithmic in the tails, defined at zero; the inverse refuses values past "
                 "overflow, where sinh leaves float64")
lego("/pre/kalfa/sinh", "scales:Sinh", alias="sinh",
     description="sinh(x / scale), the direction opposite to asinh: it stretches the tails instead of compressing "
                 "them; a value past overflow is an error, where sinh leaves float64")
lego("/pre/kalfa/tanh", "scales:Tanh", alias="tanh",
     description="tanh(x / scale) into (-1, 1); the inverse clips at 1 - eps, so a value that saturated in float64 "
                 "(past about 19 scale) comes back at the clip instead of infinity")
lego("/pre/kalfa/atanh", "scales:Atanh", alias="atanh",
     description="artanh(x / scale) of a bounded column, inverted by scale tanh(y); a value outside (-scale, scale) "
                 "is an error that names how many and how large")
