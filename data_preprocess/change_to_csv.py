f = 'occurence.txt'

target = 'occurence.csv'

target = open(target, 'w')

source = open(f, 'r')

for index, line in enumerate(source.readlines()):
    new_line = ','.join(line.split('\t'))
    source.write(new_line)
    print(index)
