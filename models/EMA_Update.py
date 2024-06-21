import torch

# class EMA:
#     def __init__(self, ema_model, main_model, decay=0.5):
#         self.ema_model = ema_model
#         self.main_model = main_model
#         self.decay = decay
#         self.shadow = {}
#
#         # 初始化影子参数
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 self.shadow[name] = param.data.clone()
#
#     def update(self):
#         for ((name, ema_param),(name, main_param)) in zip(self.ema_model.named_parameters(), self.main_model.named_parameters()):
#             assert name in self.shadow
#             new_average = (1.0 - self.decay) * main_param.data + self.decay * self.shadow[name]
#             self.shadow[name] = new_average.clone()
#
#     def apply_shadow(self):
#         for name, ema_param in self.ema_model.named_parameters():
#             assert name in self.shadow
#             ema_param.data.copy_(self.shadow[name])

def update_ema_variables(ema_model, main_model, i, alpha=0.6):
    if i == 0:
        for ema_param, main_param in zip(ema_model.parameters(), main_model.parameters()):
            ema_param.data.copy_(main_param.data)  # 初始化教师模型参数与学生模型参数相同
    else:
        alpha = min(1 - 1 / (i + 1), alpha)
        for ema_param, main_param in zip(ema_model.parameters(), main_model.parameters()):
            tmp_data = alpha * ema_param.data + (1-alpha) * main_param.data
            ema_param.data = tmp_data
            # print(ema_param==main_param)
    # for ema_param in ema_model.parameters():
    #     print(ema_param.data)