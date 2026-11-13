# LastFM dataset (HetRec 2011)

## Source
- File: hetrec2011-lastfm-2k.zip
- Downloaded from: https://grouplens.org/datasets/hetrec-2011/
- Files used: user_artists.dat (user, artist, weight) and user_friends.dat (userID, friendID)

## Original dataset paper
[zip ke andar readme.txt kholo, wahan jo citation likha hai wahi yahan copy karo]

## Raw size
- user_artists.dat: 92,834 interaction rows (92,835 lines with header)
- user_friends.dat: 25,434 friend rows (25,435 lines with header); each friendship appears in both directions

## Preprocessing in our code
[Siddiqua se confirm karo: kam se kam kitne interactions wale users aur items rakhe jaate hain]
- Column 3 (weight, number of plays) is not used. Only user and artist ids are taken (--inter-cols 0 1).
- After preprocessing: 1873 users, 6942 items, 82,113 interactions, 12,588 social edges.