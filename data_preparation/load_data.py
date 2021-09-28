from collections import Counter
import pandas as pd


#Creating the function to load in each datafile as a dataframe
def load_data(filename, category, annotations_order=list()):

    data = []
    with open (filename, encoding = 'UTF-8') as f:
        next(f)
        contents = f.readlines()
        # print(contents)

        season = 0
        episode = 0
        question = 0
        q_person = 0
        answer = 0
        a_person = 0
        annotations = 0
        goldstandard = 0

        q_and_a_original = []
        q_and_a_modified = []

        
        for idx, line in enumerate(contents):
            line = line.strip().split()

            if len(line) != 0:
                if line[0] == 'Season':
                    season = line[1]
                    episode = line[4]
            
                if line[0] == "Original:":
                    
                    q_and_a_original.append(contents[idx+1].strip())
                    q_and_a_original.append(contents[idx+2].strip())
                    
                    if len(q_and_a_original) == 2:
                        i_q_original = q_and_a_original[0].index(':')
                        i_a_original = q_and_a_original[1].index(':')

                        q_person = q_and_a_original[0][:i_q_original]
                        question_original = q_and_a_original[0][i_q_original + 2:]
                        
                        a_person = q_and_a_original[1][:i_a_original]
                        answer_original = q_and_a_original[1][i_a_original + 2:]
                
                
                if line[0] == "Modified:":
                    
                    q_and_a_modified.append(contents[idx+1].strip())
                    q_and_a_modified.append(contents[idx+2].strip())
                    
                    if len(q_and_a_modified) == 2:
                        i_q_modified = q_and_a_modified[0].index(':')
                        i_a_modified = q_and_a_modified[1].index(':')

                        question_modified = q_and_a_modified[0][i_q_modified + 2:]
                        
                        answer_modified = q_and_a_modified[1][i_a_modified + 2:] 


                elif line[0] == 'Annotation:':
                    for aidx, a in enumerate(annotations_order):
                        if a == 'g':
                            idx1 = aidx+1
                        elif a == 'b':
                            idx2 = aidx+1
                        elif a == 'p':
                            idx3 = aidx+1
                        else:
                            print('Invalid annotations, check parameters.')
                    
                    annotation_1 = line[idx1]
                    annotation_2 = line[idx2]
                    annotation_3 = line[idx3]
                    
                    annotations = [annotation_1,annotation_2,annotation_3]
                    counts = Counter(annotations)

                    if len(set(counts)) == 3:
                        goldstandard = '6'


                    else:
                        maximum = (max(counts.values()))
                        max_key = 0
                        for key, value in counts.items():
                            if value == maximum:
                                max_key = key

                        goldstandard = max_key


                    data.append([season, episode, category, q_person, a_person, question_original, answer_original, question_modified, answer_modified, annotation_1, annotation_2, annotation_3, goldstandard])
                    q_and_a_modified = []
                    q_and_a_original = []

    df = pd.DataFrame(data)
    df.columns = ['Season', 'Episode', 'Category', 'Q_person', 'A_person', 'Q_original', 'A_original', 'Q_modified', 'A_modified', 'Annotation_1', 'Annotation_2', 'Annotation_3', 'Goldstandard']
    
    return df
