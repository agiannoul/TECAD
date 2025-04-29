# from PdmContext.utils.dbconnector import SQLiteHandler
#
#
# database = SQLiteHandler(db_name=f"ContextDatabase.db")
#
# contextlist = database.get_all_context_by_target("scores")
#
#
# inters=[]
# for cont in contextlist:
#     temp=[]
#     for edge in cont.CR["edges"]:
#         if "score" in edge[0]:
#             temp.append(edge)
#     inters.append(temp)

import pickle

with open('shap_01.pickle', 'rb') as file:
    data = pickle.load(file)
    print(len(data.keys()))