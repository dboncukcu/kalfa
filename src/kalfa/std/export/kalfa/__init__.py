from kalfa.registration import pack

lego = pack(__name__)


lego("/export/kalfa/state_dict", "formats:state_dict", alias="state_dict",
     description="The model's state_dict as <stem>.pt, the plain torch weights")
lego("/export/kalfa/pt2", "formats:pt2", alias="pt2",
     description="The model exported with torch.export from one traced batch, the batch dimension left dynamic, and "
                 "saved as <stem>.pt2, the archive torch.export.load reads back")
lego("/export/kalfa/onnx", "formats:onnx", alias="onnx", requires="onnx",
     description="The model exported to <stem>.onnx from one traced batch, the wires as the input and output names, "
                 "at the opset given")
