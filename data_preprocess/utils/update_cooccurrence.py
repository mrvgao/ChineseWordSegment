import sqlite3


class Occurence:
    def __init__(self, db_name, table, force=False):
        self.table_name = table
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.c = self.conn.cursor()
        if force:
            self.c.execute('drop table if exists {}'.format('test_occ'))
        self.c.execute('''CREATE TABLE if not exists {} (word1 text, word2 text, count real)'''.format(self.table_name))

    def word_counts(self, word1, word2):
        self.c.execute("SELECT count FROM {} where word1 = '{}' and word2 = '{}'".format(self.table_name, word1, word2))
        data = self.c.fetchone()
        if data is not None:
            return data[0]
        else:
            return None

    def insert(self, word1, word2, count):
        self.c.execute('INSERT INTO {} VALUES (?, ?, ?)'.format(self.table_name), (word1, word2, count))

    def accumulate(self, word1, word2, count):
        old_count = self.word_counts(word1, word2)
        if old_count is not None:
            new_count = count + old_count
            self.c.execute("UPDATE {} SET count = {} where word1 = '{}' and word2 = '{}' ".format(self.table_name, new_count, word1, word2))
        else:
            new_count = count
            self.insert(word1, word2, count)

        return new_count

    def close(self):
        self.conn.commit()
        self.c.close()


if __name__ == '__main__':
    occ = Occurence(':memory:', 'test_occ', force=True)

    assert occ.word_counts('word1', 'word2') is None

    occ.c.execute('insert into {} values (?, ?, ?)'.format('test_occ'), ('word1', 'word2', 1.0))

    assert occ.word_counts('word1', 'word2')

    occ.insert('W1', 'W2', 1.2)

    assert occ.word_counts('W1', 'W2')

    occ.accumulate('W1', 'W2', 1.0)

    assert occ.word_counts('W1', 'W2') == 1.2 + 1.0

    occ.accumulate('new', 'new', 1.0)

    assert occ.word_counts('new', 'new') == 1.0

    occ.c.execute('select * from {}'.format(occ.table_name))
    values = occ.c.fetchall()

    print(values)

    assert len(values) > 1
    occ.close()

    occ = Occurence('test.db', 'test_occ')
    occ.c.execute('select * from {}'.format(occ.table_name))
    new_values = occ.c.fetchall()

    assert values == new_values, new_values

    print(values)
    print('test done!')


