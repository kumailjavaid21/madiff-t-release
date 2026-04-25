import re
from collections import Counter

f = open('D:/MAS_genAI/madifft_final/cas-sc-template-v2.tex', encoding='utf-8').read()
print('braces:', f.count('{'), f.count('}'))

begins = re.findall(r'\\begin\{([^}]+)\}', f)
ends = re.findall(r'\\end\{([^}]+)\}', f)
b, e = Counter(begins), Counter(ends)
mm = {k: (b[k], e[k]) for k in set(list(b) + list(e)) if b[k] != e[k]}
print('env mismatches:', mm if mm else 'none')

print('begin{figure...}:', len(re.findall(r'\\begin\{figure', f)))
print('begin{table...}:', len(re.findall(r'\\begin\{table', f)))

refs = set(re.findall(r'\\ref\{(fig:[^}]+|tab:[^}]+)\}', f))
labels = set(re.findall(r'\\label\{(fig:[^}]+|tab:[^}]+)\}', f))
print('unresolved refs:', (refs - labels) or 'none')
print('unused labels:', (labels - refs) or 'none')
print('lines:', len(f.splitlines()))

# check figure file references
figs = re.findall(r'includegraphics[^\{]*\{([^}]+)\}', f)
print('figure includes:')
for g in figs:
    print(' -', g)
