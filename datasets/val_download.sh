cd datasets
mkdir images
cd images

wget http://images.cocodataset.org/zips/val2017.zip -O coco_val2017.zip

tar -xf coco_val2017.zip
rm coco_val2017.zip

cd ..

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O coco_ann2017.zip
tar -xf coco_ann2017.zip
rm coco_ann2017.zip
