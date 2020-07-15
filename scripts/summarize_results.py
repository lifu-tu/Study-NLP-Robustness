import argparse
import glob
import json
import shutil
import os
import json
import traceback
import csv
from sklearn.metrics import classification_report, confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', nargs='+')
    parser.add_argument('--output-json')
    parser.add_argument('--error-analysis', default=None)
    args = parser.parse_args()
    return args

def get_model_config(model_path):
    res = json.load(open('{}/report.json'.format(model_path)))['config']
    return res

def analyze(path, data):
    pred_file = os.path.join('{}/predictions.tsv'.format(os.path.dirname(path)))
    with open(pred_file) as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        preds, labels = [], []
        for row in reader:
            if data == 'hans':
                preds.append('non-entailment' if row['pred'] != 'entailment' else 'entailment')
                labels.append(row['label'])
            elif data == 'swap':
                preds.append('non-contradiction' if row['pred'] != 'contradiction' else 'contradiction')
                labels.append(row['label'])
            else:
                preds.append(row['pred'])
                labels.append(row['label'])
    report = classification_report(labels, preds, output_dict=True)
    return report

def parse_file(path, error_analysis):
    #print('parsing {}'.format(path))
    try:
        res = json.load(open(path))
        config = res['config']
        model_config = get_model_config(config['init_from'])
        test_data = '{}-{}'.format(config['task_name'], config['test_split'])
        test_split = config['test_split']
        train_data = '{}-{}'.format(model_config['task_name'], model_config['test_split'])
        model_path = config['init_from'].split('/')
        model_path = '/'.join(model_path[:-1] + [model_path[-1][:5]])
        model = model_config['model_type']

        model_cheat = float(model_config['cheat'])
        test_cheat = float(config['cheat'])
        rm_cheat = float(model_config.get('remove_cheat', False))
        wdrop = float(model_config.get('word_dropout', 0))
        model = model_config.get('model_type', 'bert') or 'bert'
        superficial = model_config['superficial'] if model_config['superficial'] else '-1'
        additive = len(model_config['additive']) if model_config['additive'] else 0
        last = int(config['use_last'])
        metrics = res['test'][test_split]
        if test_data.startswith('MNLI-hans') or test_data.startswith('SNLI-swap') or test_data.startswith('MNLI-swap'):
            metric_name = 'mapped-accuracy'
        else:
            metric_name = 'accuracy'
        if additive == 0:
            acc = metrics[metric_name]
            additive = '0'
            if model_config['superficial'] is True or model_config['superficial'] == 'hypo':
                model = 'hypo'
            elif model_config['superficial'] == 'handcrafted':
                model = 'hand'
        else:
            if config['additive_mode'] == 'all':
                acc = metrics['last_{}'.format(metric_name)]
            elif config['additive_mode'] == 'last':
                acc = metrics['{}'.format(metric_name)]
            else:
                raise ValueError
            prev_models = []
            for prev in model_config['additive']:
                prev_config = get_model_config(prev)
                if prev_config['superficial'] == 'handcrafted':
                    prev_models.append('hand')
                elif prev_config['superficial']:
                    prev_models.append('hypo')
                else:
                    prev_models.append('cbow')
            additive = ','.join(prev_models)
    except Exception as e:
        traceback.print_exc()
        print(os.path.dirname(path))
        #import sys; sys.exit()
        #shutil.rmtree(os.path.dirname(path))
        return {
                'status': 'failed',
                'eval_path': path,
               }
    remove = int(model_config.get('remove', False))
    if remove == 1:
        assert not config['remove']
    report = {
            'status': 'success',
            'train_data': train_data,
            'test_data': test_data,
            'last': last,
            'mch': model_cheat,
            'tch': test_cheat,
            'rm_ch': rm_cheat,
            'sup': superficial,
            'add': additive,
            'rm': remove,
            'wdrop': wdrop,
            'model': model.upper(),
            'acc': acc,
            'model_path': model_path,
            'eval_path': path,
           }
    constraints = {
            #lambda r: r['mch'] != -1,
            #lambda r: r['tch'] == 0,
            #lambda r: r['sup'] == 0,
            #lambda r: r['add'] in ('hand', 'hypo', 'cbow', '0'),
            #lambda r: r['add'] != 'hypo,cbow',
            #lambda r: r['wdrop'] in (0,),
            #lambda r: r['rm'] in (0,),
            #lambda r: r['test_data'].startswith('MNLI-hans'),
            #lambda r: r['train_data'].startswith('MNLI'),
            #lambda r: r['test_data'] == 'SNLI-test',
            #lambda r: r['test_data'] == 'MNLI-dev_matched',
            #lambda r: r['train_data'] == 'SNLI',
            #lambda r: not r['test_data'].endswith('mismatched'),
            #lambda r: r['model'] in ('BERT', 'DA', 'ESIM'),
            #lambda r: r['model'] in ('ESIM','HYPO', 'CBOW', 'HAND'),
            #lambda r: r['model'] in ('HYPO', 'CBOW', 'HAND'),
            }
    for c in constraints:
        if not c(report):
            return {
                    'status': 'filtered',
                    'eval_path': path,
                   }
    if error_analysis is not None:
        try:
            acc_report = analyze(path, error_analysis)
            if error_analysis == 'hans':
                report.update({
                    'ent': acc_report['entailment']['f1-score'],
                    'n-ent': acc_report['non-entailment']['f1-score'],
                    'avg': acc_report['macro avg']['f1-score'],
                    'acc_report': acc_report,
                    })
            elif error_analysis == 'swap':
                report.update({
                    'con': acc_report['contradiction']['f1-score'],
                    'n-con': acc_report['non-contradiction']['f1-score'],
                    'avg': acc_report['macro avg']['f1-score'],
                    'acc_report': acc_report,
                    })
            else:
                report.update({
                    'ent': acc_report['entailment']['f1-score'],
                    'con': acc_report['contradiction']['f1-score'],
                    'neu': acc_report['neutral']['f1-score'],
                    'avg': acc_report['macro avg']['f1-score'],
                    'acc_report': acc_report,
                    })
        except Exception as e:
            traceback.print_exc()
            print(os.path.dirname(path))
            return {
                    'status': 'failed',
                    'eval_path': path,
                   }
    return report

def main(args):
    files = []
    for d in args.runs_dir:
        files.extend(glob.glob('{}/*/report.json'.format(d)))
    all_res = [parse_file(f, args.error_analysis) for f in files]
    failed_paths = [r['eval_path'] for r in all_res if r['status'] == 'failed']
    if failed_paths:
        print('failed paths:')
        for f in failed_paths:
            print(f)
        ans = input('remove failed paths? [Y/N]')
        if ans == 'Y':
            for f in failed_paths:
                shutil.rmtree(os.path.dirname(f))
            print('removed {} dirs'.format(len(failed_paths)))
        else:
            print('ignore failed paths. continue')

    all_res = [r for r in all_res if r['status'] == 'success']
    #for r in all_res:
    #    print(r['eval_path'])
    #ans = input('remove failed paths? [Y/N]')
    #if ans == 'Y':
    #    for r in all_res:
    #        shutil.rmtree(os.path.dirname(r['eval_path']))
    #import sys; sys.exit()

    columns = [
               ('train_data', 20, 's'),
               ('test_data', 50, 's'),
               #('tch', 6, '.1f'),
               #('mch', 6, '.1f'),
               #('rm_ch', 6, '.1f'),
               #('sup', 5, 's'),
               ('model', 7, 's'),
               ('rm', 5, 'd'),
               ('add', 7, 's'),
               #('wdrop', 5, '.1f'),
               ('acc', 10, '.3f'),
               #('model_path', 10, 's'),
              ]
    if args.error_analysis == 'hans':
        columns.extend([
               ('ent', 10, '.3f'),
               ('n-ent', 10, '.3f'),
               ('avg', 10, '.3f'),
            ])
    elif args.error_analysis == 'swap':
        columns.extend([
               ('con', 10, '.3f'),
               ('n-con', 10, '.3f'),
            ])
    elif args.error_analysis is not None :
        columns.extend([
               ('ent', 10, '.3f'),
               ('con', 10, '.3f'),
               ('neu', 10, '.3f'),
               ('avg', 10, '.3f'),
            ])
    #columns.append(('eval_path', 10, 's'))
    if len(all_res) == 0:
        print('no results found')
        return
    #if 'last_acc' in all_res[0]:
    #    columns.append(('last_val_acc', 10, '.2f'))
    #if 'prev_acc' in all_res[0]:
    #    columns.append(('prev_val_acc', 10, '.2f'))
    header = ''.join(['{{:<{w}s}}'.format(w=width)
                      for _, width, _ in columns])
    header = header.format(*[c[0] for c in columns])
    row_format = ''.join(['{{{c}:<{w}{f}}}'.format(c=name, w=width, f=form)
                          for name, width, form in columns])
    all_res = sorted(all_res, key=lambda x: [x[c[0]] for c in columns])

    #duplicated_paths = []
    #for i, r in enumerate(all_res):
    #    if i > 0 and r['model_path'] == all_res[i-1]['model_path']:
    #        f = r['eval_path']
    #        duplicated_paths.append(f)
    #        print(f)
    #ans = input('remove duplicated paths? [Y/N]')
    #if ans == 'Y':
    #    for f in duplicated_paths:
    #        shutil.rmtree(os.path.dirname(f))
    #import sys; sys.exit()

    #to_delete = [r['eval_path'] for r in all_res]
    #for p in to_delete:
    #    print(p)
    #ans = input('remove selected paths? [Y/N]')
    #if ans == 'Y':
    #    for f in to_delete:
    #        shutil.rmtree(os.path.dirname(f))
    #    import sys; sys.exit()


    print(header)
    for res in all_res:
        print(row_format.format(**res))

    if args.output_json:
        with open(args.output_json, 'w') as fout:
            json.dump(all_res, fout, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)
