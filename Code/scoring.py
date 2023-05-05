import torch

def get_fid_score(fid, batch_real_imgs, batch_fake_imgs):
    # convert [-1,1] range to [0,255] range
    batch_real_imgs = (batch_real_imgs + 1) * 255 / 2
    batch_fake_imgs = (batch_fake_imgs + 1) * 255 / 2

    # convert to uint8
    batch_real_imgs = batch_real_imgs.type(torch.uint8)
    batch_fake_imgs = batch_fake_imgs.type(torch.uint8)

    fid.update(batch_real_imgs, real=True)
    fid.update(batch_fake_imgs, real=False)
    
    return fid