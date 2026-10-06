"""Future table-only formatting of preserved, already aggregated summaries.

No scientific imports, fitting, predictions, metrics, profiling, statistics or
plotting. This new interface was AST checked only and was never executed during
the v10.3 source/document task. Default invocation displays a plan. Export needs
the explicit flag and a new output directory outside this packet.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent
TABLES = ('4', '5', '6', '7', '8', '9', '12', '13')


def within(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def read_exact(record):
    path = HERE / record['relative_path']
    if not within(path, HERE) or path.is_symlink() or not path.is_file():
        raise ValueError('Missing or unsafe retained summary: ' + record['relative_path'])
    data = path.read_bytes()
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Retained summary identity changed: ' + record['relative_path'])
    return json.loads(data.decode('utf-8-sig'))


def unique(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError('Duplicate retained summary key: ' + str(key))
        result[key] = row
    return result


def three(value):
    return format(value, '.3f')


def number(value):
    """Original presentation rule; formats the saved scalar without estimation."""
    if value is None:
        return r'\text{undefined}'
    if value == 0:
        return '0.000'
    if abs(value) < .01:
        mantissa, exponent = format(value, '.2e').split('e')
        return mantissa + r'\times10^{' + str(int(exponent)) + '}'
    return three(value)


def pnumber(value):
    return format(value, '.6f') if value is not None and abs(value - .05) < .0005 else number(value)


def interval(value):
    return r'\text{undefined}' if value is None else '[' + number(value[0]) + ',' + number(value[1]) + ']'


def rmse_cell(record, sd_key, scientific_small=False):
    if record['n'] != 5:
        raise ValueError('Expected an already aggregated five-seed RMSE record')
    formatter = number if scientific_small else three
    return '$' + formatter(record['mean']) + r'\pm' + formatter(record[sd_key]) + '$'


def escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def latex_table(specification, rows):
    """Presentation only: captions/column styles come from the frozen layout."""
    if specification['longtable']:
        columns = specification['column_count']
        header = specification['header']
        body = [r'\clearpage\begin{landscape}', r'\begingroup\footnotesize',
                r'\setlength{\tabcolsep}{4pt}',
                r'\begin{longtable}{' + specification['columns'] + '}',
                r'\caption{' + specification['caption'] + r'}\label{' + specification['label'] + r'}\\',
                r'\toprule ' + header + r'\\\midrule\endfirsthead',
                r'\multicolumn{' + str(columns) + r'}{l}{\tablename\ \thetable\ continued}\\',
                r'\toprule ' + header + r'\\\midrule\endhead',
                r'\midrule\multicolumn{' + str(columns) + r'}{r}{Continued on the next page}\\\endfoot',
                r'\bottomrule\endlastfoot']
        body += rows
        body += [r'\end{longtable}\endgroup', r'\end{landscape}']
    else:
        body = [r'\begin{table}[tbp]\centering\small', r'\setlength{\tabcolsep}{4pt}',
                r'\caption{' + specification['caption'] + r'}\label{' + specification['label'] + '}',
                r'\begin{tabular}{' + specification['columns'] + r'}\toprule',
                specification['header'] + r'\\\midrule', *rows,
                r'\bottomrule\end{tabular}', r'\end{table}']
    return '\n'.join(body) + '\n'


def receptor_rows(summary, specification):
    retained = unique(summary['procedures'], ('subset', 'setting', 'head'))
    tex, csv_rows = [], []
    for head, label in specification['heads']:
        cells = [label]
        row = {'head': head, 'label': label}
        for setting in specification['settings']:
            saved = retained['full', setting, head]['metrics']['rmse']
            cells.append(rmse_cell(saved, 'sample_sd'))
            row[setting + '_mean'] = saved['mean']
            row[setting + '_sample_sd'] = saved['sample_sd']
        tex.append(' & '.join(cells) + r'\\')
        csv_rows.append(row)
    return tex, csv_rows


def molecular_rows(summary, specification):
    retained = unique(summary['rows'], ('depth', 'head'))
    tex, csv_rows = [], []
    for head, label in specification['heads']:
        cells = [label]
        row = {'head': head, 'label': label}
        for depth in (1, 3):
            saved = retained.get((depth, head))
            if saved is None:
                cells += ['--', '--']
                row.update({str(depth) + '_parameters': None, str(depth) + '_mean': None, str(depth) + '_sample_sd': None})
            else:
                cells += [str(saved['parameters']), rmse_cell(saved['rmse'], 'sd', scientific_small=True)]
                row.update({str(depth) + '_parameters': saved['parameters'],
                            str(depth) + '_mean': saved['rmse']['mean'], str(depth) + '_sample_sd': saved['rmse']['sd']})
        tex.append(' & '.join(cells) + r'\\')
        csv_rows.append(row)
    return tex, csv_rows


def component_rows(summary, specification):
    retained = unique(summary['groups'], ('head',))
    tex, csv_rows = [], []
    for suffix, label in specification['procedures']:
        cells = [label]
        row = {'procedure': suffix, 'label': label}
        for order in (1, 2):
            saved = retained['m8_k' + str(order) + '_' + suffix,]
            cells += [str(saved['total_parameters']), rmse_cell(saved['rmse'], 'sample_sd')]
            row.update({str(order) + '_total_parameters': saved['total_parameters'],
                        str(order) + '_trainable_parameters': saved['trainable_parameters'],
                        str(order) + '_frozen_parameters': saved['frozen_parameters'],
                        str(order) + '_mean': saved['rmse']['mean'],
                        str(order) + '_sample_sd': saved['rmse']['sample_sd']})
        tex.append(' & '.join(cells) + r'\\')
        csv_rows.append(row)
    return tex, csv_rows


def width_rows(summary, specification):
    retained = unique(summary['groups'], ('head',))
    tex, csv_rows = [], []
    for width, tokens in specification['declared_memory_tokens']:
        error_cells, count_cells = [str(width) + ', error'], [str(width) + r', $P/L$']
        for order, count in enumerate(tokens, 1):
            head = 'm' + str(width) + '_k' + str(order) + '_learned0'
            saved = retained[head,]
            error_cells.append(rmse_cell(saved['rmse'], 'sample_sd'))
            count_cells.append('$' + str(saved['trainable_parameters']) + '/' + str(count) + '$')
            csv_rows.append({'head': head, 'width': width, 'order': order,
                             'mean': saved['rmse']['mean'], 'sample_sd': saved['rmse']['sample_sd'],
                             'trainable_parameters': saved['trainable_parameters'],
                             'declared_memory_tokens': count})
        tex += [' & '.join(error_cells) + r'\\', ' & '.join(count_cells) + r'\\[2pt]']
    return tex, csv_rows


def cost_rows(summary, specification):
    tex, csv_rows = [], []
    if len(summary['rows']) != 48:
        raise ValueError('Expected the preserved 48-row cost summary')
    for campaign, title in specification['campaigns']:
        tex.append(r'\multicolumn{7}{l}{\textbf{' + title + r'}}\\*\midrule')
        for saved in summary['rows']:
            if saved['campaign'] != campaign:
                continue
            cells = [specification['settings'][saved['setting']], specification['heads'][saved['head']],
                     '$' + number(saved['mean']) + r'\pm' + number(saved['sample_sd']) + '$',
                     format(saved['parameters'], ','), three(saved['forward_median_ms']),
                     three(saved['train_step_median_ms']), three(saved['maximum_incremental_train_MiB'])]
            tex.append(' & '.join(cells) + r'\\')
            csv_rows.append(dict(saved))
    return tex, csv_rows


def family_rows(summary, specification):
    tex, csv_rows = [], []
    for saved in summary['family_summaries']:
        family = saved['family'].split('_')[0]
        scope = specification['scopes'][family]
        tex.append(' & '.join([family, scope, str(saved['m']), str(saved['t_Holm_below_005']),
                              str(saved['sign_flip_Holm_below_005'])]) + r'\\')
        csv_rows.append(dict(family=saved['family'], scope=scope, m=saved['m'],
                             t_Holm_below_005=saved['t_Holm_below_005'],
                             sign_flip_Holm_below_005=saved['sign_flip_Holm_below_005']))
    return tex, csv_rows


def paired_rows(summary, specification):
    retained = unique(summary['results'], ('id',))
    tex, csv_rows = [], []
    for identifier, label, family in specification['contrasts']:
        saved = retained[identifier,]
        adjusted = saved['adjustments'][family]
        global_adjusted = saved['adjustments']['F00_global_union']
        if label == '__contact_label_from_retained_contrast__':
            label = 'Contact frozen: ' + saved['contrast'].replace('baseline', 'parent').replace('minus', '-').replace('pair_mlp', 'pair-MLP')
        cells = [escape(identifier + ' ' + label), family.split('_')[0], str(saved['n']),
                 '$' + number(saved['mean_difference']) + '$', '$' + interval(saved['t_reference_interval95']) + '$',
                 '$' + pnumber(saved['t_p_raw']) + '$', '$' + pnumber(adjusted['t_p_Holm']) + '$',
                 '$' + pnumber(adjusted['sign_flip_p_Holm']) + '$', '$' + pnumber(global_adjusted['t_p_Holm']) + '$']
        tex.append(' & '.join(cells) + r'\\')
        csv_rows.append(dict(id=identifier, label=label, context=saved['context'], contrast=saved['contrast'],
                             metric=saved['metric'], family=family, n=saved['n'],
                             mean_difference=saved['mean_difference'],
                             t_reference_interval95=json.dumps(saved['t_reference_interval95']),
                             t_p_raw=saved['t_p_raw'], family_t_p_Holm=adjusted['t_p_Holm'],
                             family_sign_flip_p_Holm=adjusted['sign_flip_p_Holm'],
                             global_t_p_Holm=global_adjusted['t_p_Holm']))
    return tex, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', nargs='+', choices=TABLES, default=list(TABLES))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--export-retained-tables', action='store_true')
    arguments = parser.parse_args()
    manifest = json.loads((HERE / 'retained_summary_manifest.json').read_text())
    layout = json.loads((HERE / 'presentation_spec.json').read_text())
    selected = list(dict.fromkeys(arguments.tables))
    plan = dict(scope='Formatting of already aggregated summaries only', tables=selected,
                statistics_recomputed=False, scientific_module_imports=False,
                original_analyzers_or_renderer_called=False,
                command_execution_during_v10_3=False, exact_current_full_TeX_equivalence_verified=False)
    if not arguments.export_retained_tables:
        print(json.dumps(plan, indent=2))
        return
    if arguments.output_root is None:
        raise ValueError('--output-root is required for export')
    output = arguments.output_root.resolve()
    if output.exists() or within(output, PACKET) or within(PACKET, output):
        raise ValueError('Output must be a new directory outside and not containing this packet')
    builders = {'4': receptor_rows, '5': receptor_rows, '6': molecular_rows, '7': component_rows,
                '8': width_rows, '9': cost_rows, '12': family_rows, '13': paired_rows}
    inputs, artifacts = {}, {}
    for table in selected:
        specification = layout['tables'][table]
        input_key = specification['input']
        if input_key not in inputs:
            inputs[input_key] = read_exact(manifest['inputs'][input_key])
        tex_rows, csv_rows = builders[table](inputs[input_key], specification)
        if len(csv_rows) != specification['expected_output_rows']:
            raise ValueError('Saved row selection differs from the frozen presentation plan: table ' + table)
        artifacts[table] = (latex_table(specification, tex_rows), csv_rows)
    # All input identities/schema/row selections are checked before any writes.
    output.mkdir(parents=True, exist_ok=False)
    for table, (tex, rows) in artifacts.items():
        with (output / ('table_' + table + '.tex')).open('x', encoding='utf-8') as stream:
            stream.write(tex)
        with (output / ('table_' + table + '.csv')).open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    receipt = dict(plan, output_root=str(output), retained_inputs={key:manifest['inputs'][key] for key in inputs},
                   output_csv_rows={table:len(rows) for table, (_, rows) in artifacts.items()},
                   disclosure='Reads saved aggregate fields; does not establish numerical correctness, fresh reproduction or current full-TeX equivalence.')
    with (output / 'table_formatting_receipt.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
