import csv
from utils.preprocess_gloss import process_text
def load_corpus(file_path):
    """
    コーパスのcsvからパスとテキストを読み込む
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as file:
        path_dict={}
        i=0
        reader = csv.reader(file, delimiter='|')
        for row in reader:
            if i==0:
                i+=1
                continue
            path=row[0]
            text=row[3]
            path_dict[path]=text
    return path_dict

def convert_gloss_to_class(path_pd,is_processed=False):
    """
    テキストからグロスレベルに分割，クラス番号を割り当てる(アルファベット順)
    :param path_pd(pandas):
    :return:
    """
    gloss_list=[]
    total_dict={}
    for i in path_pd["annotation"]:
        if is_processed:
            i=process_text(i)
        for g in i.split():
            if g in total_dict.keys():
                total_dict[g]+=1
            else:
                total_dict[g]=1
    gloss2class={}
    class2gloss={}
    for idx,(k,v) in enumerate(sorted(total_dict.items())):
        gloss2class[k]=idx+1
        class2gloss[idx+1]=k
    return gloss2class,class2gloss
