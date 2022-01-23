# checkpoint = torch.load("/data2/cql/code/meanT60/Checkpoints/relative_loss_addBias/t60_predict_model_73_meanT60_continue11.pt", map_location=device)
# #
# print(net.load_state_dict(checkpoint["model"], strict=False))
# print("begin train")
# checkpoint = torch.load("./Checkpoints/relative_loss_addBias/t60_predict_model_73_meanT60_continue11.pt", map_location=device)
# net.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# lr.load_state_dict(checkpoint['lr'])
# start_epoch = checkpoint['epoch'] + 1
# print("the training process from epoch{}...".format(start_epoch))