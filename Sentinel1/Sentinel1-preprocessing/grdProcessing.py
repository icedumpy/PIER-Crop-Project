import xml.etree.ElementTree as ET
import os
from osgeo import gdal

template  = "Sentinel1_50m.xml"

def grgProcessing(input_dir, output_dir):
    for r, _, f in os.walk(input_dir):
        for file in f:
            filename, file_extension = os.path.splitext(file)
            if file_extension.upper() == ".ZIP":
                input_file = os.path.join(r, file)
                print("working on {}".format(file))
                output_file = os.path.join(output_dir, filename+".tif")
                target_file = r"F:\CROP-PIER\CROP-WORK\Sentinel-1-xml-for-preprocess\Sentinel1_50m.xml"
                tree = ET.parse(target_file)
                root = tree.getroot()
                for node in root.findall("node"):
                    if node.attrib['id'] == "Read":
                        parameter = node.find("parameters")
                        file = parameter.find("file")
                        file.text = input_file

                    elif node.attrib['id'] == "Write":
                        parameter = node.find("parameters")
                        file = parameter.find("file")
                        file.text = output_file
                path_save_xml = os.path.join(output_dir, filename+'.xml')
                tree.write(path_save_xml)
                command = "gpt " + path_save_xml
                print(command)
                os.system(command)

if __name__ == "__main__":
    input_dir = r"F:\CROP-PIER\CROP-WORK\Sentinel1-new-image\raw"
    output_dir = r"F:\CROP-PIER\CROP-WORK\Sentinel1-new-image\processed"
    grgProcessing(input_dir, output_dir)
#%%
        
