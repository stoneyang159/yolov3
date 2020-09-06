import numpy as np
import torch
import PIL.Image as Image

# grid mask
# 加在 normalize 之后
def gridmask(img, label, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    if np.random.rand() > prob:
        return img, label
    h = img.size(1)
    w = img.size(2)
    d1 = 2
    d2 = min(h, w)
    hh = int(1.5 * h)
    ww = int(1.5 * w)
    d = np.random.randint(d1, d2)
    # d = self.d
    #        self.l = int(d*self.ratio+0.5)
    if ratio == 1:
        l = np.random.randint(1, d)
    else:
        l = min(max(int(d * ratio + 0.5), 1), d - 1)
    mask = np.ones((hh, ww), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    if use_h:
        for i in range(hh // d):
            s = d * i + st_h
            t = min(s + l, hh)
            mask[s:t, :] *= 0
    if use_w:
        for i in range(ww // d):
            s = d * i + st_w
            t = min(s + l, ww)
            mask[:, s:t] *= 0

    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
    mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

    mask = torch.from_numpy(mask).float()
    if mode == 1:
        mask = 1 - mask

    mask = mask.expand_as(img)
    if offset:
        offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
        offset = (1 - mask) * offset
        img = img * mask + offset
    else:
        img = img * mask

    return img, label


# mixup
def train_with_mixup(train_loader, model, criterion, optimizer, alpha):
    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = torch.from_numpy(np.array(target)).float().cuda()

        ratio = np.random.beta(alpha, alpha)

        index = torch.randperm(images.size(0)).cuda()
        inputs = ratio * images + (1 - ratio) * images[index]

        # targets_a, targets_b = target, target[index]
        targets = ratio * target + (1 - ratio) * target[index]

        outputs = model(inputs)
        # loss = ratio * criterion(outputs, targets_a) +
        # (1 - ratio) * criterion(outputs, targets_b)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# cutout
def cutout(img, n_holes, length):
    h, w = img.size(1), img.size(2)

    mask = np.ones((h, w), np.float32)

    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)  # 范围限定在[0, h]
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0 ,w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.

    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img * mask
    return img

# CutMix
def cutmix(beta):
    lam = np.random.beta(beta, beta)  # 1.0, we use uniform dist.
    rand_index = torch.randperm(input.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:,:,bbx1:bbx2, bby1:bby2] = input[rand_index, :, bby1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 -bby1) / (input.size()[-1] * input.size()[-2]))

    output = model(input)

    loss = criterion(output, target_a) * lam + \
           criterion(output, target_b) * (1. - lam)






