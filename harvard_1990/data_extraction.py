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


#### READ IN TEXT DATA

DATA_DIR = '/mnt/LINUX600GB/zimmerman_docs/'
OCR_DIR = os.path.join(DATA_DIR, 'ocr_ed/')
DATA_FILENAME = os.path.join(OCR_DIR, 'alumni_directory_1990_2.3.2022.txt')


def import_text() -> str:
    with open(DATA_FILENAME, 'r', encoding='utf-8') as fh:
        text = fh.read()
    # Cut off introductory material
    text = re.search(r'(?<=Alphabetical Roster of Alumni).+', text, flags=re.DOTALL|re.IGNORECASE).group()
    return text


def fix_common_typos(text: str) -> str:
    # fix cases where chars are substituted for numbers
    text = re.sub(r'(?<=\d)S(?=[\d ])', '5', text)
    text = re.sub(r'(?<=[\d ])€(?=[\d ])', '6', text)
    text = re.sub(r'(?<=[\dO])O(?=[\dO])', '0', text)
    text = re.sub(r'(?<=\d)B(?=\d)', '8', text)
    # fix cases where chars are substituted for letters
    text = re.sub(r'(?<=[A-Z])8(?=[A-Z])', 'B', text)
    text = re.sub(r'(?<=[A-Z])80(?=[A-Z])', 'BO', text)
    text = re.sub(r'(?<=[A-Z\-])[1l](?=[A-Z\-])', 'I', text)
    # fix state abbreviation typos
    text = re.sub(r' LIT(?= \d{5})', ' UT', text)
    text = re.sub(r' A2(?= \d{5})', ' AZ', text)
    # remove stray numbers at the start of a profile line
    text = re.sub(r'(?<=\n)\d\s*(?=[A-Z]+[,.] [A-Z])', '', text)
    # fix cases where t or f is substituted for + (dead) marker
    text = re.sub(r'(?<=[^A-Z]\s)■?[tf] ?(?=[A-Z\-ÖÄÏÜ\'’]+[,.])', '+ ', text)
    # add missing commas in case of unknown address
    text = re.sub(r'(?<=[A-Za-z]) (?=(?:Address )?Unknown?|Requested No Mail)', ', ', text)
    # miscellaneous fixes
    text = re.sub(r'(?<= [MWHR])[li](?= \d)', 'I', text)
    text = re.sub(r'(?<=\n)MC (?=[A-Z]{2})', 'MC', text)
    text = re.sub(r'(?<=\d{5})1ST ', ' IST ', text)
    text = text.replace(
        '§', 'S'
    ).replace(
        'ß', 'B'
    ).replace(
        '£', 'E'
    ).replace(
        '’', '\''
    )
    return text


def split_lines(text: str) -> str:
    # split profiles that are on the same line
    text = re.sub(r'(?<=[^\n]{20}[^\n\[]{15}(?:[^\n\[][A-Za-z ][a-zVASPE’*\']|\D\d\d)) +(?=\S??\s*[A-Z\-ÖÄÏÜ\'’]{3,}(?:,|\.)\s?[A-Z][a-z ].{,20}(?:,|\.))', '\n', text)
    text = re.sub(r'(?<=[^\n]{30}[A-Za-z\d\)][\]\)\}]) +(?=\S??\s*[A-Z\-ÖÄÏÜ\'’]{3,}(?:,|\.)\s[A-Z][a-z ].{,20}(?:,|\.))', '\n', text)
    return text


def unsplit_lines(text: str) -> str:
    # Figure out where a line break does not signify a new profile, and remove those line breaks
    # first try a couple of basic text substitutions
    text = re.sub(r'\n(?=(?:[A-Z]?[a-eg-su-z\d\]]|[A-Z][a-z\d\]]))', ' ', text)
    text = re.sub(r'\n(?=[A-Z]{1,3}\s\d\d\s)', ' ', text)
    text= re.sub(r'(?<=\[SEE)\n(?=[A-Z]{2})', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])\n(?=[IV]{1,3}(?:,|\.))', ' ', text)
    text = re.sub(r'\n(?=[A-Z]{2} \d{5})', ' ', text)
    text = re.sub(r'\n(?=\S\s*\d)', ' ', text)
    text = re.sub(r'(?<=Apt)\n(?=\S{1,4}(?:,|\.))', ' ', text)
    text = re.sub(r'\n(?=[^A-Z]?\s*(?:[^AEIOUÖÄÏÜY]{5}|[^AEIOUÖÄÏÜY,.]+[,.]))', ' ', text)
    # remove page number indicators
    text = re.sub(r'\n\d+(?:\s[A-Z\-]+)?\n', '\n', text)
    text = re.sub(r'\n[A-Z\-]+\s\d+\n', '\n', text)
    # now a more complicated procedure based on the expected first couple letters of each line
    lines_to_unsplit = []
    line_start_2 = '~~'
    line_start_1 = '~~'
    last_idx = -1
    for i, line in enumerate(text.split('\n')):
        line_start_0_search = re.search(r'.{0,3}?([A-Za-z]{2}|[A-Z].)', line)
        if line_start_0_search is None:
            # this line is out of place
            lines_to_unsplit.append(i)
            continue
        line_start_0 = line_start_0_search.group(1)
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
    text = re.sub(r'(?<=[^\n]{25}[^\n\[\(\{]{15}[A-Za-z\d\]\)\}]) (?=[^A-Za-z]?\s*(?:[A-Z\-ÖÄÏÜ\'’]{5,}|NG)(?:,|\.) (?:[A-Z][a-z]|[A-Z] [A-Z][a-z]))', '\n', text)
    text = re.sub(r'(?<=[^\n]{35}(?:(?:\(|\-)\d\d\)|hon\))) (?=[^A-Za-z]?\s*[A-Z]{2}\S*(?:,|\.) ?(?:[A-Z][a-z]{2}|[A-Z] [A-Z][a-z]))', '\n', text)
    text = re.sub(r'(?<=[^\n]{35}(?:.[A-Z]{2}|[A-Z][A-Za-z][A-Z]) \d\d) (?=[^A-Za-z]?\s*[A-Z]{2}\S*(?:,|\.) ?(?:[A-Z][a-z]|[A-Z] [A-Z][a-z]))', '\n', text)
    text = re.sub(r'(?<=[^\n]{35}(?: \d\d|\d\d\)|[A-Z][a-z]{2})) ((?:\S+ ){0,2}\S+?)[,.]?(?= (?:Ms|Dr|Prof|Miss) .{,20}[,.])', '\n\\1,', text)
    code_block = '|'.join(OCCUPATION_CODES)
    text = re.sub(r'(?<=[^\n]{25}[^\n]{15}(?:\d\d|cl| d|\d[\)\]]) )(' + code_block + r') (?=[^A-Za-z]?\s*(?:(?:\S+\s){0,2}\S+(?:,|\.) ?|[A-Z]+ )(?:[A-Z][a-z]|[A-Z] [A-Z][a-z]))', '\\1\n', text)
    text = re.sub(rf'(?<=[^,] )({code_block}) (?=\D)', '\\1\n', text)
    text = re.sub(rf'({code_block})\t\s*', '\\1\n', text)
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

# print(text)


#### NOW EXTRACT DATA FROM TEXT

# compile frequently used regexes
dead_re = re.compile(r'd(?:,|\.)?\s+(.*?\d ?\d{3})(?:,|\.)?\s*(.*)')
reported_dead_re = re.compile(r'(?:Reported Dead|[\(\[]date unknown[\)\]])(?:,|\.)\s+(.+)')
zip_re = re.compile(r'(.*?[^A-Za-z][A-Za-z] ?[A-Za-z](?:,|\.)?\s*\d{5}(?:-\d{4})?)(?:\s?(\D.*)|$)')
alt_zip1_re = re.compile(r'(.*?[^A-Za-z][A-Z]{2})\s\d{2}\S{2,}(?:\s(\D.*)|$)')
alt_zip2_re = re.compile(r'(.*?\D\d{5}(?:-\d{4})?)(?:\s(\D.*)|$)')
alt_zip3_re = re.compile(r'(.*?(?:,|\.) [A-Z]{2})\s(.*)')
house_re = re.compile(r'(?:^| )([A-Z][A-Za-z])\s+(\D.*)')
degree_re = re.compile(r'([A-Z][A-Za-z]{0,2})\s(\d{2}(?:\s?\((?:\d{2}(?:\-\d{2})? ?){1,}\))?)(?:\s([msclwhd\(\)]{2,4}))?')
occupation_re = re.compile(r'(?<=\s)[A-Z][A-Za-z]{1,2}$')
other_name_re = re.compile(r'^[\[\(\{\]](.+?)(?:[\]\)\}]|(?:[\[jlJiI](?:,|\.|$)))(?:(?:,|\.)?\s+(.+))?')
profile_re = re.compile(r'^([^A-Za-z\s]|[tf])?\s*?((?:[A-Z]+ )*[A-Z\-ÖÄÏÜ\'’a-z]+)(?:,|\.)?\s*([^,.\(\[]+)(?:,|\.)?\s*(.+)')
name_re = re.compile(r'^(\d?[A-Za-z\-ÖÄÏÜ\'’]+)(?:,|\.)?\s?(.*)')


def try_alt_zip_searches(info: str) -> typing.Tuple[str, str]:
    # try some edge cases to deal with typos
    alt_zip_search = alt_zip1_re.search(info)
    # ^ looks for case where zip code is messed up but state is intact
    if alt_zip_search is not None:
        address = alt_zip_search.group(1)
        info = alt_zip_search.group(2)
        return address, '' if info is None else info
    alt_zip_search = alt_zip2_re.search(info)
    # ^ matches case where state name is messed up but zip code is intact
    if alt_zip_search is not None:
        address = alt_zip_search.group(1)
        info = alt_zip_search.group(2)
        return address, '' if info is None else info
    alt_zip_search = alt_zip3_re.search(info)
    # ^ looks for cases where state is intact, but zip is missing
    if alt_zip_search is not None:
        address = alt_zip_search.group(1)
        info = alt_zip_search.group(2)
        return address, '' if info is None else info
    return '', info


def get_foreign_address(info: str, exclude=None) -> typing.Tuple[dict, str]:
    for c, subs in COUNTRY_REPLACEMENTS.items():
        if c in info:
            info = info.replace(c, subs)
    country = ''
    for c in COUNTRIES:
        if c != exclude and c in info:
            country = c
            break
    if not country:
        address, info = try_alt_zip_searches(info)
        if address:
            return {'address': address}, info
        # This person might actually be dead
        dead_search = dead_re.search(info)
        if dead_search is not None:
            return {'death_date': dead_search.group(1)}, dead_search.group(2)
        reported_dead_search = reported_dead_re.search(info)
        if reported_dead_search is not None:
            return {'death_date': 'unknown'}, reported_dead_search.group(1)
        # not dead but unknown country
        return {}, info
    country_search = re.search(rf'(.*{country})(?:,|\.)?(?:\s+(.*)|$)', info)
    if country_search is None or re.search(rf'[Oo]f {country}$', country_search.group(1)):
        if exclude is None:
            # maybe the wrong country was selected
            return get_foreign_address(info, country)
        else:
            return {}, info
    info = country_search.group(2)
    return {'address': country_search.group(1)}, '' if info is None else info


def get_address(info: str) -> typing.Tuple[dict, str]:
    # it's easy to identify if in the US because it ends with a zip code
    # otherwise, a bit tricky
    zip_search = zip_re.search(info)
    # ^ first group is address (ending in zip code)
    # second group is rest of info
    if zip_search is not None:
        fields = {'address': zip_search.group(1)}
        info = zip_search.group(2)
        return fields, '' if info is None else info

    # probably foreign, need more complicated approach
    fields, info = get_foreign_address(info)

    return fields, info


def make_school_data_subs(info: str) -> str:
    for regex, substitution in school_data_res:
        info = regex.sub(r'\1' + substitution, info)
    return info


def process_degree_data(degree_search: list) -> list:
    attendance = []
    for code, year, distinction in degree_search:
        fields = {
            'year': year
        }
        # figure out if it's a school code or degree code
        if code in SCHOOL_CODES:
            fields['school_code'] = code
        elif code in DEGREE_CODES:
            fields['degree_code'] = code
        else:
            fields['degree_code'] = code + '?'
        # add distinction if extant
        if distinction:
            fields['distinction'] = distinction
        attendance.append(fields)
    return attendance
            

def get_school_data(info: str) -> dict:
    fields = {}
    # first make substitutions to fix common typos
    info = make_school_data_subs(info)
    # now look for house code
    house_search = house_re.search(info)
    if house_search is not None:
        house_code = house_search.group(1).capitalize()
        if house_code in HOUSE_CODES:
            fields['house_code'] = house_code
        else:
            fields['house_code'] = house_code + '?'
        info = house_search.group(2)
    # next look for information about degrees (or school code for non-completers)
    degree_search = degree_re.findall(info)
    # ^ first field is code, next field is year(s), last field is distinctions
    if degree_search:
        fields['attendance'] = process_degree_data(degree_search)
    # next look for occupation code
    occupation_search = occupation_re.search(info)
    if occupation_search is not None:
        occupation = occupation_search.group()
        if occupation not in OCCUPATION_CODES:
            occupation += '?'
        fields['occupation_code'] = occupation
    return fields

def process_info_from_line(info: str, is_living: bool) -> dict:
    original = info  # keep a copy around in case we need to revert to it
    fields = {'notes': []}
    # Check for case where it says [SEE other name]
    other_name_search = other_name_re.search(info)
    # ^ first matching group is alternate name
    # second group is rest of profile
    if other_name_search is not None:
        within_search = re.search(
            r'(SEE\s*)?(.+)',
            other_name_search.group(1)
        )
        fields['alternate_name'] = within_search.group(2)
        if other_name_search.group(2) is None:
            # nothing else here, just a link to married name profile
            fields['notes'].append('is_maiden_name')
            return fields
        # now we can move on to extracting rest of information
        info = other_name_search.group(2)
    if is_living:
        # if living, first field should be address
        new_fields, info = get_address(info)
        if new_fields == {}:
            # maybe they're actually dead?
            if dead_re.search(info) is not None:
                return process_info_from_line(original, False)
            # else
            fields['notes'].append('had_error')
        fields.update(new_fields)
    else:
        # if dead, first field should be death date
        death_search = dead_re.search(info)
        if death_search is None:
            # sometimes it just says "Reported Dead"
            reported_dead_search = reported_dead_re.search(info)
            if reported_dead_search is None:
                # often this just means that this person is not actually dead
                return process_info_from_line(original, True)
            else:
                fields['death_date'] = 'unknown'
                info = reported_dead_search.group(1)
        else:
            fields['death_date'] = death_search.group(1)
            info = death_search.group(2)
    fields.update(get_school_data(info))
    return fields


def process_line(line: str) -> dict:
    """Takes a line from the book and returns a dict of fields"""
    profile_search = profile_re.search(line)
    # ^ first matching group is indicator for dead
    # second group is last name
    # third group is first and middle names
    # fourth group is rest of profile
    if profile_search is None:
        # not a profile
        return {}
    is_living = (profile_search.group(1) is None)
    fields = {
        'raw': line,
        'last': profile_search.group(2),
        'first': profile_search.group(3),
    }
    fields.update(process_info_from_line(profile_search.group(4).replace('*', ''), is_living))
    return fields


def get_datum(line: str, last_line: str = '') -> typing.Tuple[dict, str]:
    if last_line:
        datum = process_line(f'{last_line} {line}')
        new_datum = process_line(line)
        if not datum and not new_datum:
            return {}, ''
        if new_datum and 'had_error' not in new_datum['notes']:
            datum = new_datum
        elif not datum:
            return {}, ''
    else:
        datum = process_line(line)
        if not datum:
            return {}, ''
    return datum, line if 'had_error' in datum['notes'] else ''


def merge_wives(data: list) -> list:
    # seperate first and last maiden names for married women in data
    for person in data:
        if person.get('alternate_name') is None or 'is_maiden_name' in person['notes']:
            continue
        name_search = name_re.search(person['alternate_name'])
        if name_search is None:
            person['notes'].append('had_error')
            continue
        person['married_last'] = person['last']
        person['last'] = name_search.group(1)
        person['married_first'] = person['first']
        person['first'] = name_search.group(2)
        del person['alternate_name']
    # remove profiles that contain only maiden names
    return [x for x in data if 'is_maiden_name' not in x['notes']]


def process_all(lines: list) -> list:
    data = []
    last_line = ''
    last_datum = {}
    for line in lines:
        try:
            datum, new_last_line = get_datum(line, last_line)
            if last_datum and (new_last_line or not last_line):
                has_problem = (
                    (last_datum.get('house_code') and last_datum['house_code'].endswith('?'))
                    or (
                        last_datum.get('attendance') and any(
                            x['degree_code'].endswith('?') for x in last_datum['attendance'] if x.get('degree_code')
                        )
                    )
                    or (last_datum.get('occupation_code') and last_datum['occupation_code'].endswith('?'))
                )
                if has_problem and 'had_error' not in last_datum['notes']:
                    last_datum['notes'].append('had_error')
                data.append(last_datum)
            last_line = new_last_line
            last_datum = datum
        except (AttributeError, TypeError, ValueError) as e:
            print(f'ERROR: {e} in "{line}"')
    if last_datum:
        data.append(last_datum)
    return merge_wives(data)


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

    with open(os.path.join(DATA_DIR, 'data_1990.json'), 'w', encoding='utf-8') as fh:
        json.dump(data, fh)

    # for datum in data:
    #     if 'had_error' in datum['notes'] and 'SEE' not in datum['raw']:
    #         print(datum['raw'])

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
    
    occupations_not_found = []
    for d in data:
        code = d.get('occupation_code')
        if code and code.endswith('?'):
            occupations_not_found.append(code)
        
    c_occ = Counter(occupations_not_found)
    print(c_occ.most_common())

    error_count = len([d for d in data if 'had_error' in d['notes']])
    n_data =  len(data)
    print(f'Error rate: {error_count} / {n_data} = {error_count / n_data:.4f}')

    # print('\n'.join([d['raw'] for d in data if d.get('house_code') == f'{args[1]}?']))


    # print('\n'.join([d['raw'] for d in data if d.get('attendance') is not None and any(a['degree_code'] == 'MA?' for a in d['attendance'] if a.get('degree_code'))]))
