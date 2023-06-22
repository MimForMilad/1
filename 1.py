import pandas as pd

movies_df = pd.read_csv('movies.dat', sep='::', engine='python', header=None, names=['MovieID', 'Title', 'Genres'])

print(movies_df.head())

#output:

#  MovieID                               Title                        Genres
#0        1                    Toy Story (1995)   Animation|Children's|Comedy
#1        2                      Jumanji (1995)  Adventure|Children's|Fantasy
#2        3             Grumpier Old Men (1995)                Comedy|Romance
#3        4            Waiting to Exhale (1995)                  Comedy|Drama
#4        5  Father of the Bride Part II (1995)                        Comedy

users_df = pd.read_csv('users.dat', sep='::', engine='python', header=None, names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])

print(users_df.head())
#Output:
#   UserID Gender  Age  Occupation Zip-code
#0       1      F    1          10    48067
#1       2      M   56          16    70072
#2       3      M   25          15    55117
#3       4      M   45           7    02460
#4       5      M   25          20    55455

merged_df = pd.merge(movies_df, movies_df, on='MovieID')

print(merged_df.head())

#Output:
#   MovieID             Title                       Genres  UserID  Rating  Timestamp
#0        1  Toy Story (1995)  Animation|Children's|Comedy       1       5  978824268
#1        1  Toy Story (1995)  Animation|Children's|Comedy       6       4  978237008
#2        1  Toy Story (1995)  Animation|Children's|Comedy       8       4  978233496
#3        1  Toy Story (1995)  Animation|Children's|Comedy       9       5  978225952
#4        1  Toy Story (1995)  Animation|Children's|Comedy      10       5  978226474
