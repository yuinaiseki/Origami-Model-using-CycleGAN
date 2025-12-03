def nst(content_path, style_path, obj_name, output_path=None,
        num_steps=NUM_STEPS, 
        style_weight=STYLE_WEIGHT, 
        content_weight=CONTENT_WEIGHT,
        alpha=LEARNING_RATE,
        config_name=ACTIVE_LAYER_CONFIG,
        metric_callback=None,
        output_dir=None,):
    """core NST implementation: logs numbers (e.g. loss, avg time taken) each step + image results in specified dir
    
    Args:
        content_path: Either a file path (str) or a PyTorch tensor
        style_path: Either a file path (str) or a PyTorch tensor
    """
    
    start_time = time.time()
    
    # get config
    config = LAYER_CONFIGS[config_name]
    content_layers = config["content"]
    style_layers = config["style"]
    style_layer_weights = config['style_weights']

    # model
    vgg_model = vgg19(pretrained=True).features
    for param in vgg_model.parameters():
        param.requires_grad_(False)
    vgg_model.to(device).eval()

    print("loading images")
    # imgs, in tensor format - handle both paths and tensors
    if isinstance(content_path, str):
        content_tensor = img_to_tensor(content_path)
    else:
        content_tensor = content_path
        
    if isinstance(style_path, str):
        style_tensor = img_to_tensor(style_path)
    else:
        style_tensor = style_path

    print("getting features")
    # extract
    content_features = extract_features(content_tensor, content_layers, model=vgg_model)
    style_features = extract_features(style_tensor, style_layers, model=vgg_model)

    print("calculating grams")
    # calculate gram
    style_grams = {layer: gram_matrix(style_features[layer]) 
                   for layer in style_layers}
    
    result = content_tensor.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([result], lr=alpha)

    loss_history = {
        'total': [],
        'content': [],
        'style': [],
        'step': []
    }

    if output_dir:
            final_output_dir = os.path.join(output_dir, f'{obj_name}')
            os.makedirs(final_output_dir, exist_ok=True)
            log_file = os.path.join(final_output_dir, f'training_log_{obj_name}.txt')
            with open(log_file, 'w') as f:
                f.write(f"NST training log\n")
                f.write(f"{'='*80}\n")
                f.write(f"config: {config_name}\n")
                f.write(f"content: {content_path if isinstance(content_path, str) else 'tensor'}\n")
                f.write(f"style: {style_path if isinstance(style_path, str) else 'tensor'}\n")
                f.write(f"total steps: {num_steps}\n")
                f.write(f"content weight: {content_weight}\n")
                f.write(f"style weight: {style_weight}\n")
                f.write(f"learning rate: {alpha}\n")
                f.write(f"{'='*80}\n\n")

            print(f"started logging to: {log_file}")

    start_time = time.time()
    last_log_time = start_time

    for step in range(num_steps):
        result_features = extract_features(result, content_layers+style_layers, model=vgg_model)

        # content
        content_loss = 0
        for layer in content_layers:
            content_loss += torch.mean((result_features[layer] - content_features[layer])**2)
        # avg loss for all layers
        content_loss = content_loss / len(content_layers)

        style_loss = 0
        for layer in style_layers:
            result_feature = result_features[layer]
            gram = gram_matrix(result_feature)
            style_gram = style_grams[layer]

            layer_weight = style_layer_weights.get(layer, 1.0)
            
            batch, channels, height, width = result_feature.shape
            layer_style_loss = layer_weight * torch.mean(
                (gram - style_gram)**2
            )
            style_loss += layer_style_loss / (channels * height * width)

        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_history['total'].append(total_loss.item())
        loss_history['content'].append(content_loss.item())
        loss_history['style'].append(style_loss.item())
        loss_history['step'].append(step)
        
        if metric_callback:
            metric_callback(step, total_loss.item(), content_loss.item(), style_loss.item(), result)

        if step % 100 == 0:
            current_time = time.time()
            cumulative_time = current_time - start_time
            step_time = current_time - last_log_time
            
            # console
            print(f"step {step:4d}/{num_steps} | "
                f"total loss: {total_loss.item():10.2f} | "
                f"content loss: {content_loss.item():8.4f} | "
                f"style loss: {style_loss.item():10.6f} | "
                f"time taken (cumulative): {cumulative_time:6.2f}s | " 
                f"time taken (~100 steps): {step_time:6.2f}s")
            
            # Save image
            if output_dir:
                intm_img = tensor_to_img(result)
                intm_path = os.path.join(final_output_dir, f'{obj_name}_step_{step:04d}.png')
                intm_img.save(intm_path)
                
                # Append to log file
                with open(log_file, 'a') as f:
                    f.write(f"{step:<8} {total_loss.item():<15.2f} "
                        f"{content_loss.item():<15.4f} {style_loss.item():<15.6f} "
                        f"{step_time:<12.2f}\n")
                    
            last_log_time = current_time

    end_time = time.time()
    total_time = end_time - start_time

    final_img = tensor_to_img(result)
    if output_dir:
        final_path = os.path.join(final_output_dir, f'{obj_name}_final.png')
        final_img.save(final_path)
        print(f"saved final result in {final_path}")
        
        # Append final summary to log
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL RESULTS!!\n")
            f.write(f"{'='*80}\n")
            f.write(f"total time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n")
            f.write(f"avg time per step: {total_time/num_steps:.3f}s\n")
            f.write(f"final total loss: {loss_history['total'][-1]:.2f}\n")
            f.write(f"final content loss: {loss_history['content'][-1]:.4f}\n")
            f.write(f"final style loss: {loss_history['style'][-1]:.6f}\n")
        
        print(f"training log saved in {log_file}")
    
    print(f"time taken: {total_time:.2f}s")
    
    return final_img