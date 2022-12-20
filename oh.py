
 
from time import time, sleep
from os import listdir
 
# TODO: 1. 

#Sets start time
start_time = time()
#Replace sleep(75) below with code you want to time
sleep(75)
#Sets end time
end_time = time()
#Computes overall runtime in seconds
tot_time = end_time - start_time
#Prints overall runtime in seconds
print("\nTotal Elapsed Runtime : ", tot_time,"in seconds.")  
#definition，主要是定义宠物图像文件夹，模型名称（后面要比较不同的模型,所以模型名称会被修改），小狗名称定义文件的名称
classify_dir='pet_image/'
arch='vgg'
dogfile='dognames.txt'
 
 
hours = int( (tot_time / 3600) )
minutes = int( ( (tot_time % 3600)/ 60 ) )
seconds = int ( ( (tot_time % 3600)% 60 ) )


print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600)) ) + ":"+
str( int( ( (tot_time % 3600)/ 60 ) ) ) +":"+
str( int(( (tot_time % 3600)% 60 ) ) ) )

# TODO: 2. 

def get_pet_labels():
    pic_repo_addr = "/home/aistudio/pet_image/"
    pic_repo = listdir(pic_repo_addr)
    dataset = dict()
    for pic in pic_repo:
        pet_name = pic[:pic.rfind("_")].lower()
        dataset[pic] = pet_name
    return dataset
answers_dic = get_pet_labels()


#Imports only listdir function from oS module
from os import listdir
# Retrieve the filenames from folder pet_images/
filename_list = listdir("pet_images/")
#Print 10 of the filenames from folder pet_images/
print(" \nPrints 10 filenames from folder pet_images/")
for idx in range(0,10,1):
    print("%2d file: %-25s" % (idx + 1, filename_list[idx]))
    

 #Sets pet_image variable to a filename
pet_image = "Boston_terrier_02259.jpg"
#Sets string to lower case letters
low_pet_image = pet_image.lower()
#Splits lower case string by _ to break into words
word_list_pet_image = low_pet_image.split("_")
#Create pet_name starting as empty string
pet_name =""
#Loops to check if word in pet name is only
#alphabetic characters - if true append word
# to pet_name separated by trailing space
for word in word_list_pet_image:
    if word.isalpha() :
        pet_name += word + ""
# Strip off starting/trailing whitespace characters
pet_name = pet_name.strip()
#Prints resulting pet_name
print("\nFilename=", pet_image," Label=", pet_name)   

    
    
    
    
# TODO: 3. 
#使用本模块定义classify_images函数
def classify_images(images_dir, petlabel_dic, model):
   """
    使用classifier函数得到40张图片的分类器分类结果，并和前面的图片label进行比较，
    最后创建一个字典包含所有的labels以及比较的结果，并将该字典结果返回。
    参数: 
       images_dir - 完整的文件夹路径
       petlabel_dic - 包含了宠物图片label的字典，它的key是宠物文件名，它的值是宠物图片label。
       model - 使用的模型名称: resnet alexnet vgg (string)
    返回值:
       results_dic - 本函数生成的新的字典，key是宠物文件名字，值是一个列表 
             (index)idx 0 = 宠物图片label (string)
                    idx 1 = 分类器label (string)
                    idx 2 = 1/0 (int) 1 =两个label一致 and 0 = 两个label不一致
   """
   results_dic = dict()
   for image in listdir(images_dir):
      pred = classifier(images_dir+image, model)
      res = [petlabel_dic[image], pred, petlabel_dic[image]==pred]
      results_dic[image] = res
   return results_dic


  
results_dic = classify_images(classify_dir, answers_dic, arch)
 
import paddlehub as hub
 
resnet_v2_50_imagenet = hub.Module("resnet_v2_50_imagenet")
alexnet_imagenet = hub.Module("alexnet_imagenet")
vgg19_imagenet = hub.Module("vgg19_imagenet")
 
#定义好的模型调用的名称
models = {'resnet': resnet_v2_50_imagenet, 'alexnet': alexnet_imagenet, 'vgg': vgg19_imagenet}
 
def classifier(image_path, model):
  """
    该函数通过加载Paddlehub的预训练模型，调用模型的classification函数，并得到图片的分类结果。
    参数: 
      image_path - 需要识别的图片的完整路径
      model -使用这个参数指定预训练的模型，模型值为以下三种: resnet alexnet vgg (string)
    返回值:
      result - 该图片分类结果label
  """
  model = models[model]
  res = model.classification(data={"image":[image_path]})
  return list(res[0][0].keys())[0].lower().replace(" ", "_")  
    
    
    
# TODO: 4. 
def adjust_results4_isadog(results_dic, dogsfile):
       """
    调整结果字典results_dic的内容，通过和dognames.txt的内容进行比较，找到哪些图片是小狗，哪些图片不是，并标记出来
    为后面的统计数据做准备
    参数:
      results_dic - 结果字典，键key是图片文件名，值是一个列表:
             (index)idx 0 = 宠物的图片Label (string)
                    idx 1 = 分类器给出的label(string)
                    idx 2 = 1/0 (int)  1 ：图片label和分类器label相等  0 = 两个label不相等
                    ---  idx 3 & idx 4 是本函数增加的内容 ---
                    idx 3 = 1/0 (int)  1 = 图片label是小狗  0 = 图片label不是小狗 
                    idx 4 = 1/0 (int)  1 = 分类器label是小狗 0 = 分类器label不是小狗
      dogsfile - 一个包含1000种label的txt文件，里面包含了ImageNet数据集中所有出现过的狗狗种类。
                这个文件里每行都有一个小狗种类.
    返回值:
           None 
       """           
       with open(dogsfile, "r") as f:
           dogs = [dog.lower() for dog in f.read().split()]
       for image in results_dic:
              label, pred, idx2 = results_dic[image]
              results_dic[image] = [label, pred, idx2, label in dogs, pred in dogs]
       return
 
results_dic = classify_images(image_dir, get_pet_labels(), "resnet")
adjust_results4_isadog(results_dic, "dognames.txt")

adjust_results4_isadog(results_dic, dogfile)
 
# TODO: 5.
def calculates_results_stats(results_dic):
    """
    这个函数用于对results_dic中的数据进行统计。
    参数:
      results_dic - 结果字典，键key是图片文件名，值是一个列表:
             (index)idx 0 = 宠物的图片Label (string)
                    idx 1 = 分类器给出的label(string)
                    idx 2 = 1/0 (int)  1 ：图片label和分类器label相等  0 = 两个label不相等
                    ---  idx 3 & idx 4 是本函数增加的内容 ---
                    idx 3 = 1/0 (int)  1 = 图片label是小狗  0 = 图片label不是小狗 
                    idx 4 = 1/0 (int)  1 = 分类器label是小狗 0 = 分类器label不是小狗
    返回值:
      results_stats - 统计结果字典，键是统计的类型，值是统计的结果。
    """
    A = 0
    B = 0
    C = 0
    D = 0
    E = 0
    Y = 0
    Z = 0
    for image in results_dic:
      Z += 1
      result = results_dic[image]
      if result[3] and result[4]:
        A += 1
      if result[3]:
        B += 1
      if not result[3] and not result[4]:
        C += 1
      if not result[3]:
        D += 1
      if result[3] and result[2]:
        E += 1
      if result[2]:
        Y += 1
    results_stats = {"n_images": Z, "n_correct_dogs": A, "n_dogs": B, "pct_correct_dogs": A/B*100, "n_correct_not_dogs": C, "n_not_dogs": D, 
                     "pct_correct_not_dogs": C/D*100, "n_correct_breed": E, "pct_correct_breed": E/B*100, "n_correct_pred": Y, "pct_correct_pred": Y/Z*100}
    return results_stats
 
  
results_stats = calculates_results_stats(results_dic)
print(results_stats)
results_stats_dic = calculates_results_stats(results_dic)
 
# TODO: 6. 
def print_results(results_dic, result_stats, model, print_incorrect_dogs, print_incorrect_breed):
       """
       展示我们前面工作的结论
       参数:
       results_dic - 结果字典，键key是图片文件名，值是一个列表:
              (index)idx 0 = 宠物的图片Label (string)
                     idx 1 = 分类器给出的label(string)
                     idx 2 = 1/0 (int)  1 ：图片label和分类器label相等  0 = 两个label不相等
                     ---  idx 3 & idx 4 是本函数增加的内容 ---
                     idx 3 = 1/0 (int)  1 = 图片label是小狗  0 = 图片label不是小狗 
                     idx 4 = 1/0 (int)  1 = 分类器label是小狗 0 = 分类器label不是小狗
       results_stats - 统计结果字典，键是统计的类型，值是统计的结果。
       model - 预训练模型名称: resnet alexnet vgg (string)
       print_incorrect_dogs - True- 展示没有正确分类的图片名字 False - 不做展示（bool） 
       print_incorrect_breed - True- 展示没有正确分类的狗狗品种 False - 不做展示 (bool) 
       返回值:
              None - 没有返回值，本函数用于打印最终的统计展示.
       """    
       print("# Model: ", model)
       print("# Total images: ", results_stats["n_images"])
       print("# Dog images: ", result_stats["n_dogs"])
       print("# Not dog images: ", result_stats["n_not_dogs"])
       print("@ Accuracy of Dog: %.1f%%" % result_stats["pct_correct_dogs"])
       print("@ Accuracy of Breed: %.1f%%" % result_stats["pct_correct_breed"])
       print("@ Accuracy of Not Dog: %.1f%%" % result_stats["pct_correct_not_dogs"])
       print("@ Accuracy of Prediction: %.1f%%" % result_stats["pct_correct_pred"])
 
       if print_incorrect_dogs:
              count = 1
              print("-"*30+"Incorrect Dogs"+"-"*30+"\n")
              for image in results_dic:
                     if results_dic[image][3] and not results_dic[image][2]:
                            print("| {:0>2d}. {}".format(count, image))
                            count += 1
              print("-"*74)
 
       if print_incorrect_breed:
              count = 1
              print("-"*30+"Incorrect Breed"+"-"*29+"\n")
              for image in results_dic:
                     if results_dic[image][3] and not results_dic[image][2]:
                            print("| {:0>2d}. {}".format(count, image))
                            count += 1
              print("-"*74)
                            
 
print_results(results_dic, results_stats, "resnet", True, True)
print_results(results_dic, results_stats, arch, True, True)
 
# TODO: 1. 输出最终运行时长格式为 hh:mm:ss 
end_time = time()
 
run_time = int(end_time-start_time)
hh = run_time // 3600
mm = (run_time % 3600)//60
ss = run_time % 60
print("Run time: {:0>2d}:{:0>2d}:{:0>2d}.".format(hh, mm, ss))
 
check_time = time()
sleep(1)
if 0.9 < time()-check_time < 1.1:
    print("Time module run successfully.")
else:
    print("Time module run failed.")

   