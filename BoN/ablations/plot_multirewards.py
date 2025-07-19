# # # # import json
# # # # import matplotlib.pyplot as plt

# # # # with open('./multireward_uncond.json') as json_data:
# # # #     data_uncond = json.load(json_data)
# # # #     json_data.close()
    
# # # # exp_rew_1_uncond = []
# # # # exp_rew_2_uncond = []

# # # # for i in data_uncond.keys():
# # # #     exp_rew_1_uncond.append(data_uncond[i]['exp_rew_1'])
# # # #     exp_rew_2_uncond.append(data_uncond[i]['exp_rew_2'])
    
# # # # with open('./multireward_code4.json') as json_data:
# # # #     data_code4 = json.load(json_data)
# # # #     json_data.close()
    
# # # # exp_rew_1_code4 = []
# # # # exp_rew_2_code4 = []

# # # # for i in data_code4.keys():
# # # #     exp_rew_1_code4.append(data_code4[i]['exp_rew_1'])
# # # #     exp_rew_2_code4.append(data_code4[i]['exp_rew_2'])
    
# # # # with open('./multireward_code40.json') as json_data:
# # # #     data_code40 = json.load(json_data)
# # # #     json_data.close()
    
# # # # exp_rew_1_code40 = []
# # # # exp_rew_2_code40 = []

# # # # for i in data_code40.keys():
# # # #     exp_rew_1_code40.append(data_code40[i]['exp_rew_1'])
# # # #     exp_rew_2_code40.append(data_code40[i]['exp_rew_2'])
    
# # # # with open('./multireward_freedom.json') as json_data:
# # # #     data_freedom = json.load(json_data)
# # # #     json_data.close()
    
# # # # exp_rew_1_freedom = []
# # # # exp_rew_2_freedom = []

# # # # for i in data_freedom.keys():
# # # #     exp_rew_1_freedom.append(data_freedom[i]['exp_rew_1'])
# # # #     exp_rew_2_freedom.append(data_freedom[i]['exp_rew_2'])
    
# # # # with open('./multireward_codex.json') as json_data:
# # # #     data_codex = json.load(json_data)
# # # #     json_data.close()
    
# # # # exp_rew_1_codex = []
# # # # exp_rew_2_codex = []

# # # # for i in data_codex.keys():
# # # #     exp_rew_1_codex.append(data_codex[i]['exp_rew_1'])
# # # #     exp_rew_2_codex.append(data_codex[i]['exp_rew_2'])
    
# # # # plt.scatter(exp_rew_2_uncond,exp_rew_1_uncond,marker='x',label='Base_SD',color='black')
# # # # plt.scatter(exp_rew_2_code4,exp_rew_1_code4,marker='o',label='CoDe (N=4)',color='orange')
# # # # plt.scatter(exp_rew_2_code40,exp_rew_1_code40,marker='o',label='CoDe (N=40)',color='green')
# # # # plt.scatter(exp_rew_2_freedom,exp_rew_1_freedom,marker='o',label='FreeDoM',color='red')
# # # # plt.scatter(exp_rew_2_codex,exp_rew_1_codex,marker='o',label='CoDeX',color='blue')
# # # # plt.ylabel("Aesthetic Score")
# # # # plt.xlabel("PickScore")  
# # # # plt.grid()
# # # # plt.legend()
# # # # plt.title("Multireward Aesthetic v/s Pickscore")
# # # # plt.savefig('multireward.jpg')

# # # import json
# # # import matplotlib.pyplot as plt

# # # def load_rewards(filepath):
# # #     with open(filepath) as f:
# # #         data = json.load(f)
# # #     rew1 = [v['exp_rew_1'] for v in data.values()]
# # #     rew2 = [v['exp_rew_2'] for v in data.values()]
# # #     return rew1, rew2

# # # # Load all datasets
# # # exp_rew_1_uncond, exp_rew_2_uncond = load_rewards('./multireward_uncond.json')
# # # exp_rew_1_code4, exp_rew_2_code4 = load_rewards('./multireward_code4.json')
# # # exp_rew_1_code40, exp_rew_2_code40 = load_rewards('./multireward_code40.json')
# # # exp_rew_1_freedom, exp_rew_2_freedom = load_rewards('./multireward_freedom.json')
# # # exp_rew_1_codex, exp_rew_2_codex = load_rewards('./multireward_codex.json')

# # # # Plotting with improved styles
# # # plt.figure(figsize=(10, 7))
# # # plt.scatter(exp_rew_2_uncond, exp_rew_1_uncond, marker='x', label='Base_SD', color='black', s=80)
# # # plt.scatter(exp_rew_2_code4, exp_rew_1_code4, marker='o', label='CoDe (N=4)', edgecolors='orange', facecolors='none', s=80)
# # # plt.scatter(exp_rew_2_code40, exp_rew_1_code40, marker='s', label='CoDe (N=40)', edgecolors='green', facecolors='none', s=80)
# # # plt.scatter(exp_rew_2_freedom, exp_rew_1_freedom, marker='^', label='FreeDoM', edgecolors='red', facecolors='none', s=80)
# # # plt.scatter(exp_rew_2_codex, exp_rew_1_codex, marker='D', label='CoDeX', edgecolors='blue', facecolors='none', s=80)

# # # plt.xlabel("PickScore", fontsize=14)
# # # plt.ylabel("Aesthetic Score", fontsize=14)
# # # plt.grid(True)
# # # plt.legend(fontsize=12)
# # # plt.xticks(fontsize=12)
# # # plt.yticks(fontsize=12)
# # # plt.tight_layout()
# # # plt.savefig('multireward.jpg', dpi=300)
# # # plt.show()

# # import json
# # import matplotlib.pyplot as plt

# # def load_rewards(filepath):
# #     with open(filepath) as f:
# #         data = json.load(f)
# #     rew1 = [v['exp_rew_1'] for v in data.values()]
# #     rew2 = [v['exp_rew_2'] for v in data.values()]
# #     return rew1, rew2

# # # Load all datasets
# # exp_rew_1_uncond, exp_rew_2_uncond = load_rewards('./multireward_uncond.json')
# # exp_rew_1_code4, exp_rew_2_code4 = load_rewards('./multireward_code4.json')
# # exp_rew_1_code40, exp_rew_2_code40 = load_rewards('./multireward_code40.json')
# # exp_rew_1_freedom, exp_rew_2_freedom = load_rewards('./multireward_freedom.json')
# # exp_rew_1_codex, exp_rew_2_codex = load_rewards('./multireward_codex.json')

# # # Plotting with solid markers and black outlines
# # plt.figure(figsize=(10, 7))
# # plt.scatter(exp_rew_2_uncond, exp_rew_1_uncond, marker='x', label='Base_SD', color='black', s=80)
# # plt.scatter(exp_rew_2_code4, exp_rew_1_code4, marker='o', label='CoDe (N=4)', facecolors='orange', edgecolors='black', s=80)
# # plt.scatter(exp_rew_2_code40, exp_rew_1_code40, marker='s', label='CoDe (N=40)', facecolors='green', edgecolors='black', s=80)
# # plt.scatter(exp_rew_2_freedom, exp_rew_1_freedom, marker='^', label='FreeDoM', facecolors='red', edgecolors='black', s=80)
# # plt.scatter(exp_rew_2_codex, exp_rew_1_codex, marker='D', label='CoDeX', facecolors='blue', edgecolors='black', s=80)

# # plt.xlabel("PickScore", fontsize=14)
# # plt.ylabel("Aesthetic Score", fontsize=14)
# # plt.grid(True)
# # plt.legend(fontsize=16)
# # plt.xticks(fontsize=16)
# # plt.yticks(fontsize=16)
# # plt.tight_layout()
# # plt.savefig('multireward_2.jpg', dpi=300)
# # plt.show()

# import json
# import matplotlib.pyplot as plt

# def load_rewards(filepath):
#     with open(filepath) as f:
#         data = json.load(f)
#     rew1 = [v['exp_rew_1'] for v in data.values()]
#     rew2 = [v['exp_rew_2'] for v in data.values()]
#     return rew1, rew2

# # Load all datasets
# exp_rew_1_uncond, exp_rew_2_uncond = load_rewards('./multireward_uncond.json')
# exp_rew_1_code4, exp_rew_2_code4 = load_rewards('./multireward_code4.json')
# exp_rew_1_code40, exp_rew_2_code40 = load_rewards('./multireward_code40.json')
# exp_rew_1_freedom, exp_rew_2_freedom = load_rewards('./multireward_freedom.json')
# exp_rew_1_codex, exp_rew_2_codex = load_rewards('./multireward_codex.json')

# # Plotting with solid markers and black outlines
# plt.figure(figsize=(10, 7))
# plt.scatter(exp_rew_2_uncond, exp_rew_1_uncond, marker='x', label='Base_SD', color='black', s=120)
# plt.scatter(exp_rew_2_code4, exp_rew_1_code4, marker='o', label='CoDe (N=4)', c='orange', edgecolors='black', s=120)
# plt.scatter(exp_rew_2_code40, exp_rew_1_code40, marker='s', label='CoDe (N=40)', c='green', edgecolors='black', s=120)
# plt.scatter(exp_rew_2_freedom, exp_rew_1_freedom, marker='^', label='FreeDoM', c='red', edgecolors='black', s=120)
# plt.scatter(exp_rew_2_codex, exp_rew_1_codex, marker='D', label='CoDeX', c='blue', edgecolors='black', s=120)

# plt.xlabel("PickScore", fontsize=18)
# plt.ylabel("Aesthetic Score", fontsize=18)
# plt.grid(True)
# plt.legend(fontsize=16)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title("Multireward Aesthetic vs PickScore", fontsize=20)
# plt.tight_layout()
# plt.savefig('multireward_2.jpg', dpi=300)
# plt.show()

import json
import matplotlib.pyplot as plt

def load_rewards(filepath):
    with open(filepath) as f:
        data = json.load(f)
    rew1 = [v['exp_rew_1'] for v in data.values()]
    rew2 = [v['exp_rew_2'] for v in data.values()]
    return rew1, rew2

# Load datasets
exp_rew_1_uncond, exp_rew_2_uncond = load_rewards('./multireward_uncond.json')
exp_rew_1_code4, exp_rew_2_code4 = load_rewards('./multireward_code4.json')
exp_rew_1_code40, exp_rew_2_code40 = load_rewards('./multireward_code40.json')
exp_rew_1_freedom, exp_rew_2_freedom = load_rewards('./multireward_freedom.json')
exp_rew_1_codex, exp_rew_2_codex = load_rewards('./multireward_codex.json')

# Colorblind-friendly colors
colors = {
    "Base_SD": "#000000",       # Black
    "CoDe4": '#009E73',         # Orange-Brown
    "CoDe40": '#E69F00',        # Sky Blue
    "FreeDoM": '#F0E442',       # Bluish Green
    "CoDeX": '#56B4E9',         # Vermillion
}


# Plot
plt.figure(figsize=(10, 7))
plt.scatter(exp_rew_2_uncond, exp_rew_1_uncond, marker='x', label='Base_SD', color=colors["Base_SD"], s=160)
plt.scatter(exp_rew_2_code4, exp_rew_1_code4, marker='o', label='CoDe (N=4)', c=colors["CoDe4"], edgecolors='black', s=120)
plt.scatter(exp_rew_2_code40, exp_rew_1_code40, marker='s', label='CoDe (N=40)', c=colors["CoDe40"], edgecolors='black', s=120)
plt.scatter(exp_rew_2_freedom, exp_rew_1_freedom, marker='^', label='FreeDoM', c=colors["FreeDoM"], edgecolors='black', s=120)
plt.scatter(exp_rew_2_codex, exp_rew_1_codex, marker='D', label='UniCoDe', c=colors["CoDeX"], edgecolors='black', s=120)

plt.xlabel("PickScore", fontsize=18)
plt.ylabel("Aesthetic Score", fontsize=18)
plt.title("Multireward Aesthetic vs PickScore", fontsize=20)
plt.grid(True)
plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('multireward_colorblind_safe.jpg', dpi=300)
plt.show()
