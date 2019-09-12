def denormalization(in_img):
    out_img = (in_img+1.0)/2.0
    return out_img.clamp(min=0.0, max=1.0)

