from PIL import Image

original = Image.open("butterfly2.jpg")
mask = Image.open("butterfly2_mask.png")

original = original.resize(mask.size)

mask = mask.convert("RGB")

result = Image.new("RGB", original.size)

for x in range(original.width):
    for y in range(original.height):
        original_pixel = original.getpixel((x, y))
        mask_pixel = mask.getpixel((x, y))
        
        if mask_pixel == (255, 255, 255):
            result.putpixel((x, y), original_pixel)
        else:
            result.putpixel((x, y), (0, 0, 0))

# Save the result
result.save("butterfly2_masked.png")