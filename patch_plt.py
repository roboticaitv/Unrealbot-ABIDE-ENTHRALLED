import glob

files = glob.glob('AADFBS_*.py') + glob.glob('AADFBS_*.PY')
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    content = content.replace('plt.show()', '# plt.show()')
    with open(f, 'w') as file:
        file.write(content)
    print(f"Patched {f}")
