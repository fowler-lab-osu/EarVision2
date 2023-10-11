'''
Not a necessary part of earvision
'''

imageListFile = "xml_files_inByear.txt"
joseListFile = "joseimg.txt"

imageList = []
joseList = []

with open(imageListFile, "r") as imageFile:
    for line in imageFile:
        lineparts = line.split('/')
        imgname = lineparts[-1].split(".")
        imageList.append(imgname[0].strip())
with open(joseListFile, "r") as joseFile:
    for line in joseFile:
        joseList.append(line.strip())

print("images in jose but not in john's list")
for img in joseList:
    if img not in imageList:
        print(img)

print()
print("images in john's list but not in jose inference")
for img in imageList:
    if img not in joseList:
        print(img)
