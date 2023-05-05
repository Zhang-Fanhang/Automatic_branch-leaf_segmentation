"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'branch': [0, 1]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_part_seg_msg', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    TEST_DATASET = PartNormalDataset(root = root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=0)
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    checkpoint = torch.load(str(experiment_dir) + '/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        seen_class = [0,0]
        correct_class = [0,0]

        Prec_0 = []
        Rec_0 = []
        F1_0 = []

        Prec_1 = []
        Rec_1 = []
        F1_1 = []

        Prec = []
        Rec = []
        F1 = []

        IoU_0 = []
        IoU_1 = []

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
        seg1 = 0
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            point_sample = points.reshape(2048, 3)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred
            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            correct = np.sum(cur_pred_val == target)

            np.savetxt("./predict/{}_sampled.txt".format(batch_id),
                       np.hstack((point_sample.cpu().data.numpy(), target.reshape(-1, 1))), fmt='%f', delimiter=' ')
            np.savetxt("./predict/{}.txt".format(batch_id),
                       np.hstack((point_sample.cpu().data.numpy(), cur_pred_val.reshape(-1, 1))), fmt='%f',
                       delimiter=' ')
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                seen_class[l] = np.sum(target == l)
                total_seen_class[l] += np.sum(target == l)
                correct_class[l] = (np.sum((cur_pred_val == l) & (target == l)))
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))




            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                IoU_0.append(part_ious[0])
                IoU_1.append(part_ious[1])
                shape_ious[cat].append(np.mean(part_ious))

            # Single result
            pred_0 = 2048-np.count_nonzero(segp)
            pred_1 = np.count_nonzero(segp)

            prec_0 = correct_class[0]/pred_0
            rec_0 = correct_class[0] / seen_class[0]

            prec_1 = correct_class[1]/pred_1
            rec_1 = correct_class[1] / seen_class[1]

            f1_0 = 2*prec_0*rec_0/(prec_0+rec_0)
            f1_1 = 2*prec_1*rec_1/(prec_1+rec_1)

            Prec_0.append(prec_0)
            Rec_0.append(rec_0)
            F1_0.append(f1_0)

            Prec_1.append(prec_1)
            Rec_1.append(rec_1)
            F1_1.append(f1_1)

            Prec.append((prec_1+prec_0)/2)
            Rec.append((rec_0+rec_1)/2)
            F1.append((f1_1+f1_0)/2)


            seg1 += np.count_nonzero(segp)



        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        # print(shape_ious)
        print(shape_ious)
        test_metrics['accuracy'] = total_correct / float(total_seen)

        print(np.array(total_correct_class))
        print(np.array(total_seen_class, dtype=np.float))

        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            # print(len(cat))
            # print(shape_ious[cat])
            # print(cat + ' ' * (14 - len(cat)))
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
        print(all_shape_ious)
    # print(mean_shape_ious)
    # print(all_shape_ious)
    print(seg1)
    mre_1 = np.array(total_correct_class)[0]/np.array(total_seen_class)[0]
    mre_2 = np.array(total_correct_class)[1]/np.array(total_seen_class)[1]
    mre = (mre_1+mre_2)/2
    mpre_1 = np.array(total_correct_class)[0]/(2048*len(TEST_DATASET)-seg1)
    mpre_2 = np.array(total_correct_class)[1]/seg1
    mpre = (mpre_1 + mpre_2) / 2

    print(mre_1,mre_2,mpre_1,mpre_2)
    f1 = 2*mre*mpre/(mre+mpre)
    log_string('Accuracy is: %.5f'%test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f'%test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
    log_string('Inctance avg mIOU is: %.5f'%test_metrics['inctance_avg_iou'])
    log_string('m_recall is: %.5f'%mre)
    log_string('m_precison is: %.5f'%mpre)
    log_string('F1-Score is: %.5f'%f1)

    log_string('---------------')
    log_string('Prec is: %.5f'%(np.mean(np.array(Prec))))
    print(Prec)
    log_string('Rec is: %.5f'%(np.mean(np.array(Rec))))
    print(Rec)
    log_string('F1 is: %.5f'%(np.mean(np.array(F1))))
    print(F1)
    log_string('IoU is: %.5f' % (np.mean(np.array(all_shape_ious))))
    print(all_shape_ious)
    log_string('Prec_0 is: %.5f'%(np.mean(np.array(Prec_0))))
    print(Prec_0)
    log_string('Rec_0 is: %.5f'%(np.mean(np.array(Rec_0))))
    print(Rec_0)
    log_string('F1_0 is: %.5f'%(np.mean(np.array(F1_0))))
    print(F1_0)
    log_string('IoU_0 is: %.5f' % (np.mean(np.array(IoU_0))))
    print(IoU_0)
    log_string('Prec_1 is: %.5f'%(np.mean(np.array(Prec_1))))
    print(Prec_1)
    log_string('Rec_1 is: %.5f'%(np.mean(np.array(Rec_1))))
    print(Rec_1)
    log_string('F1_1 is: %.5f'%(np.mean(np.array(F1_1))))
    print(F1_1)
    log_string('IoU_1 is: %.5f' % (np.mean(np.array(IoU_1))))
    print(IoU_1)

    np.savetxt("./result/{}.txt".format("OverAll"), np.array([np.mean(np.array(Prec)),np.mean(np.array(Rec)),np.mean(np.array(F1)),np.mean(np.array(all_shape_ious))]), fmt='%f', delimiter=' ')

    np.savetxt("./result/{}.txt".format("Prec"),np.array(Prec).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("Prec_0"),np.array(Prec_0).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("Prec_1"),np.array(Prec_1).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("Rec"),np.array(Rec).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("Rec_0"),np.array(Rec_0).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("Rec_1"),np.array(Rec_1).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("F1"),np.array(F1).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("F1_0"),np.array(F1_0).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("F1_1"),np.array(F1_1).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("IoU"),np.array(all_shape_ious).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("IoU_0"),np.array(IoU_0).reshape((-1,1)),fmt='%f',delimiter=' ')
    np.savetxt("./result/{}.txt".format("IoU_1"),np.array(IoU_1).reshape((-1,1)),fmt='%f',delimiter=' ')









if __name__ == '__main__':
    args = parse_args()
    main(args)

