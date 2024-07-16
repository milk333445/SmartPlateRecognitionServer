# from lprtools.convert import ModelConverter
from lprtools.lprtools.convert import ModelConverter


if __name__ == '__main__':
    config_path = 'configs/convert.yaml'
    model_converter = ModelConverter(config_path)
    model_converter.convert_onnx_to_tensorrt()
