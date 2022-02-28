from multiprocessing.sharedctypes import Value
import os
import re
import json
import typing

import multiprocessing  # who needs C++ when you have loads of CPU cores? ;P

#### IMPORT FILES TO BE USED LATER

with open('country_replacements.json', 'r') as fh:
    COUNTRY_REPLACEMENTS = json.load(fh)

with open('countries.json', 'r') as fh:
    COUNTRIES = json.load(fh)

with open('house_codes.json', 'r') as fh:
    HOUSE_CODES = json.load(fh)

with open('school_codes.json', 'r') as fh:
    SCHOOL_CODES = json.load(fh)

with open('degree_codes.json', 'r') as fh:
    DEGREE_CODES = json.load(fh)

with open('occupation_codes.json', 'r') as fh:
    OCCUPATION_CODES = json.load(fh)

school_data_res = []
with open('school_data_substitutions.json', 'r') as fh:
    subs = json.load(fh)
    for key, value in subs.items():
        school_data_res.append(
            (re.compile(rf'( |^){key}(?= |$|,)'), value)
        )


DATA_DIR = '/mnt/LINUX600GB/zimmerman_docs/'
OCR_DIR = os.path.join(DATA_DIR, 'ocr_ed/')
DATA_FILENAME = os.path.join(OCR_DIR, 'alumni_directory_1965_1.27.2022.txt')


OCC_HOUSE_GROUP = '(?:' + '|'.join(OCCUPATION_CODES + HOUSE_CODES) + ')'
OCC_GROUP = '(?:' + '|'.join(OCCUPATION_CODES) + ')'
SCHOOL_DEG_GROUP = '(?:' + '|'.join(SCHOOL_CODES + DEGREE_CODES) + ')'
HONORS_GROUP = '|'.join(['\\(hon\\)', 'cl', 'mcl', 'scl', 'w', '\\(s\\)'])

FIRST_NAME = r'(?:(?:Mrs|Dr) )?(?:[A-Z][a-zä]+ )?[A-Z][a-zä]+(?:[,.] | \()'
STATES = r'(?:Oregon|Ohio|Texas|Idaho|Puerto Rico|Conn|Maine|Florida|Virgin Islands|Alaska|Hawaii|Utah|Mass|Wi)'


def import_text() -> str:
    with open(DATA_FILENAME, 'r', encoding='utf-8') as fh:
        text = fh.read()
    return text


def fix_common_typos(text: str) -> str:
    # fix shifted punctuation marks
    text = re.sub(r'(?<=[A-Za-z]) (,.)(?=[A-Za-z])', '\\1 ', text)
    # get rid of periods after titles; not really a typo, just helps with consistency
    text = re.sub('(?<= )(Mrs|Dr|[A-Z])\.', '\\1', text)
    # fix digit/letter mix-ups
    text = re.sub(r'(?<=[A-Z])(\d) (?=\d\-)', ' \\1', text)
    text = re.sub(r'(?<=[a-z])l ?(?=\d[\-\s\(,.])', '1', text)
    text = re.sub(r'll(?=[\(\-]\d)', '11', text)
    text = re.sub(r'(?<=[A-Za-z])[lI]O(?=\-\d)', '10', text)
    text = re.sub(r'(?<=[a-z])Ol(?=\-\d)', '01', text)
    text = re.sub(r'OO(?=\-\d\d)', '00', text)
    text = re.sub(r'(?<=\s)o(?=\d\d)', 'c', text)
    text = re.sub(r'(?<=\d\d\-)ll', '11', text)
    text = re.sub(r'll(?=\-\d\d)', '11', text)
    text = re.sub(r'(?<=[A-Za-z])I(?= ?\d\-)', '1', text)
    text = re.sub(r'(?<=\s)U( ?\d\-)', 'L 1', text)
    text = re.sub(r'(?<=\s)[sS]O(?=\d\-)', 's0', text)
    # some edits relating to degree codes, etc.
    text = re.sub(r'(?<=[\d\-])O(?=[\d\-])', '0', text)
    text = re.sub(r'(?<=\s)SP(?= ?\d\d)', 'sp', text)
    text = re.sub(r'(?<=\s)0(?=\d\d[d\(\)\-]+)', 'c ', text)
    text = re.sub(r'(?<=\d\d)r\-?(?=\d\d)', '-', text)
    text = re.sub(r'([A-Z]{2,})([smwcl]{1,3})', '\\1 \\2', text)
    text = re.sub(r'(?<= )([cw])l ?(?=\d\-)', '\\1 1', text)
    text = re.sub(r'(?<=\s)C(?=\d\d)', 'c', text)
    text = re.sub(r' ?\(Hon\) ?', ' (hon) ', text)
    text = re.sub(r'¿>?(?=\d\d)', 'gb', text)
    text = re.sub(r'¿(?=\d\D)', 'g5', text)
    text = re.sub(r'(?<=\s)R[Cc](?=\d\d)', 'rc', text)
    text = re.sub(r'(?<=\s)Rg(?=\d\d)', 'rg', text)
    text = re.sub(r'(?<=\s)En&(?=\s)', 'Eng', text)
    text = re.sub(r'Ed\.? ?Adm', 'EdAdm', text)  # scrunch these together just for convenience
    # misc punctuation edits
    text = re.sub(r'(?<=[A-Za-z][^,.])(?= see Mrs)', ',', text)
    text = re.sub(r'[;:](?= )', ',', text)
    # removing some marking characters that OCR has trouble with
    text = re.sub(r'[■♦•*]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    # some more misc edits
    text = re.sub('Jamaica Plain', 'JAMAICA PLAIN', text)  # so this doesn't get picked up as country of jamaica
    text = text.replace('..', '.,').replace('’', '\'').replace('~', '-')
    return text


def split_lines(text: str) -> str:
    # split profiles that are on the same line
    text = re.sub(rf'(?<=. |\d\d)({OCC_HOUSE_GROUP}|{HONORS_GROUP})[^A-Za-z\d]? [^A-Za-z]? ?(?=[A-Z][a-zä]+\s*[,.] {FIRST_NAME})', '\\1\n', text)
    text = re.sub(rf'(?<![EWNS] |.\d)((?: |\d\d){SCHOOL_DEG_GROUP})[^A-Za-z\d]? [^A-Za-z]? ?(?=[A-Z][a-zä]+\s*[,.] {FIRST_NAME})', '\\1\n', text)
    return text


def unsplit_lines(text: str) -> str:
    # Figure out where a line break does not signify a new profile, and remove those line breaks
    # first try a basic text substitutions
    text = re.sub(r'\n(?!(?:[A-Z][A-Za-z\']*[ \-]|d[ie]l? ?|von )*[A-Z][A-Za-z\']*[,.])', ' ', text)
    text = re.sub(rf'\n(?={OCC_HOUSE_GROUP}\. [A-Za-z][a-z]* ?\d\d)', ' ', text)
    lines_to_unsplit = []
    line_start_2 = '~~'
    line_start_1 = '~~'
    last_idx = -1
    for i, line in enumerate(text.split('\n')):
        line_start_0_search = re.search(r'.{0,3}?([A-Za-zä]{2}|[A-Z].)', line)
        if line_start_0_search is None:
            # this line is out of place
            lines_to_unsplit.append(i)
            continue
        line_start_0 = line_start_0_search.group(1).lower()
        # figure out if previous line is out of place

        if i > 1 and (
            (line_start_2[0] == line_start_0[0] and line_start_1[0] != line_start_2[0])
            or (line_start_2 == line_start_0 and line_start_1 != line_start_2)
        ):
            lines_to_unsplit.append(last_idx)
        last_idx = i
        line_start_2 = line_start_1
        line_start_1 = line_start_0
    # ...remove new line chars according to lines_to_unsplit...
    newline_indices = [i for i, x in enumerate(text) if x == '\n']
    text_list = list(text)
    for line_idx in lines_to_unsplit:
        text_list[newline_indices[line_idx-1]] = ' '
    text = ''.join(text_list)
    return text


def split_lines2(text: str) -> str:
    # do another pass to split things that shouldn't have been combined
    text = re.sub(rf'(?<=[^,][^\d]\d\d|.\d\d\))( {SCHOOL_DEG_GROUP})? (?=\S{{2,}}\s*[,.] [A-Z][a-zä]+,? )', '\n', text)
    return text


def remove_line_junk(text: str) -> str:
    # remove page numbers that end up on end of line
    text = re.sub(r'(?: \d{1,4} [A-Z]{4,}| [A-Z]{4,} \d{1,4})(?=\n)', '', text)
    # remove noise from end of line
    text = re.sub(r'[^A-Za-z\d\(\)\[\]\{\}]+(?=\n)', '', text)
    return text


def preprocess_text(text: str) -> str:
    # combine all above procedures
    return remove_line_junk(
        split_lines2(
            unsplit_lines(
                split_lines(
                    fix_common_typos(
                        text
                    )
                )
            )
        )
    )


def parallel_preprocess_text(text: str) -> str:
    """Just helpful for making things go faster"""
    # split into smaller subtexts
    lines = text.split('\n')
    nlines = len(lines)
    ngroups = min(20, multiprocessing.cpu_count())
    lines_per_group = nlines // ngroups
    texts = [
        '\n'.join(lines[i*lines_per_group:(i+1)*lines_per_group]) for i in range(ngroups)
    ]
    texts[-1] += '\n'.join(lines[(ngroups+1)*lines_per_group:])
    # now just pass in parallel through text processing function
    with multiprocessing.Pool(len(texts)) as pool:
        texts = pool.map(preprocess_text, texts)
    text = '\n'.join(texts)
    return text


#### NOW EXTRACT DATA FROM TEXT

# compile frequently used regexes
dead_re = re.compile(r'd(?:,|\.)?\s+(.*?\d ?\d{3})(?:,|\.)?\s*(.*)')
reported_dead_re = re.compile(r'(?:Reported Dead|[\(\[]date unknown[\)\]])(?:,|\.)\s+(.+)')
zip_re = re.compile(rf'^(.*?(?:[A-Z][a-z]{{1,4}}|[A-Z]\.? ?[A-Z]|, {STATES})(?:,|\.)?,?\s*\d{{5}}(?:-\d{{4}})?) ?(.*)')
zip2_re = re.compile(rf'^(.{{5,}}?(?:[A-Z][a-z]{{1,4}}[,.]|[A-Z]\.? ?[A-Z]\.?|, {STATES})),? ((?:{OCC_HOUSE_GROUP}[,.]?\s|{SCHOOL_DEG_GROUP}[\s\d]).*)')
house_re = re.compile(r'(?:^| )([A-Z][A-Za-z])[,.]?\s+(\D.*)')
degree_re = re.compile(rf'([A-Za-z]+) ?({HONORS_GROUP})? ?([\d\(][\d\(\)\- ]+) ?(?:([A-Za-z]+) \D)?({HONORS_GROUP})?(?:[ ,.]|$)')
occupation_re = re.compile(rf'(?:^|[^,] )({OCC_GROUP})[,.]? (?!\d|{HONORS_GROUP})')
see_other_re = re.compile(r'^see\s')
profile_re = re.compile(r'^[^A-Z]?((?:[A-Z][A-Za-z\'ä]*[ \-]|d[ie]l? ?|von )*[A-Z][A-Za-z\'ä]*)[,.]\s*(.+?(?:, Jr\.?[.,]?)?(?:[,.] \d\d)?[,.\)]) ?(.*)')
first_re = re.compile(r'^(.+?(?:, Jr\.?[.,]?)?)(?: \(([A-Z][a-zä].*)\))?(?: (\S?\d\d))?[,.]?$')
name_re = re.compile(r'^(\d?[A-Za-zä\-ÖÄÏÜ\'’]+)(?:,|\.)?\s?(.*)')

country_res = [re.compile(rf'(.*{country})(?:(?:,|\.|\s)\s*(.*)|[,.]?$)') for country in COUNTRIES]


def get_foreign_address(info: str, exclude=None) -> typing.Tuple[dict, str]:
    for c, subs in COUNTRY_REPLACEMENTS.items():
        if c in info:
            info = info.replace(c, subs)
    country = ''
    for i, c in enumerate(COUNTRIES):
        if c != exclude and c in info:
            country = c
            break
    if not country:
        return {}, info
    country_search = country_res[i].search(info)
    if country_search is None or re.search(rf'(?:[Oo]f |New |[A-Za-z]){country}$', country_search.group(1)):
        if exclude is None:
            # maybe the wrong country was selected
            return get_foreign_address(info, country)
        else:
            return {}, info
    info = country_search.group(2)
    return {'address': country_search.group(1).strip(', ')}, '' if info is None else info


def get_address(info: str) -> typing.Tuple[dict, str]:
    # it's easy to identify if in the US because it ends with a zip code
    # otherwise, a bit tricky
    zip_search = zip_re.search(info)
    # ^ first group is address (ending in zip code)
    # second group is rest of info
    if zip_search is not None and re.search(r'Box \d{5}$', zip_search.group(1)):
        # solve problem where it sometimes picks up PO boxes
        zip_search = re.search('^(.*?[A-Z]\.? ?[A-Z](?:,|\.)?,?\s*\d{5}(?:-\d{4})?) ?(.*)', info)
    if zip_search is not None:
        fields = {'address': zip_search.group(1).strip(', ')}
        info = zip_search.group(2)
        return fields, '' if info is None else info
    # try a different search
    zip_search = zip2_re.search(info)
    if zip_search is not None:
        fields = {'address': zip_search.group(1).strip(', ')}
        info = zip_search.group(2)
        return fields, '' if info is None else info
    # look for foreign address
    return get_foreign_address(info)


def make_school_data_subs(info: str) -> str:
    for regex, substitution in school_data_res:
        info = regex.sub(rf'\1{substitution}', info)
    return info


def process_degree_data(degree_search: list) -> dict:
    fields = {}
    attendance = []
    for code1, distinction1, year, code2, distinction2 in degree_search:
        degree_fields = {}
        if code1 in SCHOOL_CODES:
            degree_fields['school_code'] = code1
        elif not code2 and code1 in DEGREE_CODES:
            degree_fields['degree_code'] = code1
        else:
            degree_fields['school_code'] = code1 + '?'
            fields['had_error'] = 1
        # add distinction if extant
        if distinction1 or distinction2:
            degree_fields['distinction'] = ' '.join((d for d in (distinction1, distinction2) if d))
        # second code should be degree name if extant
        if code2:
            if code2 in DEGREE_CODES:
                degree_fields['degree_code'] = code2
            elif re.search(HONORS_GROUP, code2) and not degree_fields.get('distinction'):
                degree_fields['distinction'] = code2
            else:
                degree_fields['degree_code'] = code2 + '?'
                fields['had_error'] = 1
        # finally year
        if year:
            degree_fields['year'] = year.strip()
        attendance.append(degree_fields)
    fields['attendance'] = attendance
    return fields
            

def get_school_data(info: str) -> dict:
    fields = {}
    # separate degree elements
    info = re.sub(r'(?<=[A-Za-z])(?=[\d\(])', ' ', info)
    info = re.sub(r'(?<=[\d\)])(?=[A-Za-z])', ' ', info)
    # make substitutions to fix common typos
    info = make_school_data_subs(info)
    # now look for house code
    house_search = house_re.search(info)
    if house_search is not None:
        house_code = house_search.group(1).capitalize()
        if house_code in HOUSE_CODES:
            fields['house_code'] = house_code
            info = house_search.group(2)
    # look for occupation code
    occupation_search = occupation_re.search(info)
    if occupation_search is not None:
        occupation = occupation_search.group(1)
        # I used OCC_GROUP in regex, so guaranteed to be an occupation code
        # if occupation not in OCCUPATION_CODES:
        #     occupation += '?'
        #     fields['had_error'] = 1
        fields['occupation_code'] = occupation
    # next look for information about degrees (or school code for non-completers)
    degree_search = degree_re.findall(info)
    # ^ first field is code, next field is distinction, next field is year(s), last is distinctions again
    if degree_search:
        fields.update(process_degree_data(degree_search))
    return fields


def get_dead_data(info: str) -> dict:
    fields = {}
    dead_search = re.search(r'(?:(?:^| )([A-Za-z]{1,3})[,.]? )?died (\d.{,15}\d{4})', info)
    if not dead_search:
        fields['had_error'] = 1
        return fields
    fields['death_date'] = dead_search.group(2)
    school_code = dead_search.group(1)
    if school_code and school_code in SCHOOL_CODES:
        fields['school_code'] = school_code
    elif school_code:
        fields['had_error'] = 1
    return fields


def process_info_from_line(info: str) -> dict:
    fields = {}
    # Check for case where it says see other name
    is_profile_redirect = see_other_re.search(info) is not None
    if is_profile_redirect:
        return {}
    # first check if this person is dead
    if 'died' in info:
        fields.update(get_dead_data(info))
    else:
        # first field should be address
        new_fields, info = get_address(info)
        if new_fields == {} and len(info) > 30:
            # no address found; if there's a lot of info left, that's probably a mistake
            fields['had_error'] = 1
        fields.update(new_fields)
        fields.update(get_school_data(info))
    return fields


def process_line(line: str) -> dict:
    """Takes a line from the book and returns a dict of fields"""
    profile_search = profile_re.search(line)
    # first group is last name
    # second group is first name and some misc info
    # third group is rest of profile
    if profile_search is None:
        # not a profile
        return {}
    # look for misc info
    first_text = profile_search.group(2).rstrip(',.')
    first_search = first_re.search(first_text)
    fields = {
        'raw': line,
        'last': profile_search.group(1),
        'first': first_search.group(1).strip(',.'),
    }
    other_name = first_search.group(2)
    if other_name:
        fields['other_name'] = other_name
    class_year = first_search.group(3)
    if class_year:
        fields['class_year'] = class_year
    fields.update(process_info_from_line(profile_search.group(3)))
    return fields


def process_all(lines: list) -> list:
    data = []
    for line in lines:
        try:
            datum = process_line(line)
            if datum:
                data.append(datum)
        except (AttributeError, TypeError, ValueError) as e:
            print(f'ERROR: {e} in "{line}"')
    return data


def parallel_process_all(lines: list) -> list:
    cores = min(multiprocessing.cpu_count(), 10)
    lines_per_core = len(lines) // (cores - 1)
    line_allocs = (
        [lines[i*lines_per_core:(i+1)*lines_per_core] for i in range(cores-1)]
        + [lines[(cores-1)*lines_per_core:]]
    )
    with multiprocessing.Pool(cores) as pool:
        lines_out = pool.map(process_all, line_allocs)
    return sum(lines_out, [])



if __name__ == "__main__":

    text = parallel_preprocess_text(import_text())

    data = parallel_process_all(text.split('\n'))

    with open(os.path.join(DATA_DIR, 'data_1965.json'), 'w', encoding='utf-8') as fh:
        json.dump(data, fh)

    # print(json.dumps(data, indent=1))

    from collections import Counter

    house_codes_not_found = []
    for d in data:
        code = d.get('house_code')
        if code and code.endswith('?'):
            house_codes_not_found.append(code)

    c_house = Counter(house_codes_not_found)
    print(c_house.most_common())

    
    degree_codes_not_found = []
    for d in data:
        attendance = d.get('attendance')
        if not attendance:
            continue
        for item in attendance:
            degree_code = item.get('degree_code')
            if degree_code and degree_code.endswith('?'):
                degree_codes_not_found.append(degree_code)

    c_deg = Counter(degree_codes_not_found)
    print(c_deg.most_common())

    error_count = len([d for d in data if d.get('had_error')])
    n_data =  len(data)
    print(f'Error rate: {error_count} / {n_data} = {error_count / n_data:.4f}')
