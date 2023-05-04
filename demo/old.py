def colission_solver(self, outputs):
        classes = outputs.pred_classes.to('cpu').numpy().astype(int)
        masks = outputs.pred_masks
        for i in range(6):
            if i in classes:
                if i == 3:
                    if 5 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(5)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask )
                if i == 0:
                    if 5 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(5)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 3 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(3)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                if i == 1:
                    if 5 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(5)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 3 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(3)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                    if 0 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(0)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                if i == 2:
                    if 5 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(5)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 3 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(3)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                    if 0 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(0)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 1 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(1)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                if i ==4:
                    if 5 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(5)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 3 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(3)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask) 
                    if 0 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(0)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 1 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(1)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
                    if 2 in classes:
                        tmp_mask = torch.logical_and(masks[list(classes).index(i)], masks[list(classes).index(2)])
                        masks[list(classes).index(i)] = torch.logical_xor(masks[list(classes).index(i)], tmp_mask)
        outputs.pred_masks = masks
        return outputs