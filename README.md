# Steam Analysis
Matthew Epland  
[phy.duke.edu/~mbe9](http://www.phy.duke.edu/~mbe9)  

## Cloning the Repository
ssh  
```bash
git clone git@github.com:mepland/steam_ana.git
```

https  
```bash
git clone https://github.com/mepland/steam_ana.git
```
## Installing Dependencies
It is recommended to work in a `virtualenv` to avoid clashes with other installed software. A useful extension for this purpose is [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/). Follow the instructions in the documentation to install and initialize wrapper before continuing.  

```bash
mkvirtualenv newenv
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
```

## Getting the Data
### Download and Extract
Download `steam.sql.gz` from [steam.internet.byu.edu](http://steam.internet.byu.edu/), then extract with gunzip. Note that the download size is 18GB and the extracted size is 161GB so you'll probably need a decent workstation.

```bash
wget http://steam.phoenixteam.net/steam.sql.gz
gunzip steam.sql.gz
```

### Setup MySQL and Load the SQL Dump
```bash
# setup user and database
mysql -u root -p
GRANT ALL PRIVILEGES ON *.* TO 'user'@'localhost' IDENTIFIED BY 'pw';
mysql -u user -password=pw
CREATE DATABASE steamdb; USE steamdb;

# load the SQL dump
SET autocommit=0;
source steam.sql;
COMMIT;
```

### Extract the Needed Data to CSV
```bash
# First look
mysql -u user --password=pw --database=steamdb --execute='SELECT steamid, appid, playtime_forever FROM Games_1 WHERE playtime_forever > 120 LIMIT 50;' -q -n -B -r > test_out.tsv

# save values out to create graph (top 5 games per user, each having at least 120 minutes of play time)
# https://www.databasejournal.com/features/mysql/selecting-the-top-n-results-by-group-in-mysql.html
# raw command:

SELECT steamid, appid, playtime_forever
 FROM
 (
   SELECT steamid, appid, playtime_forever,
   @steamid_rank := IF(@current_steamid = steamid,
                         @steamid_rank + 1,
                         1
                      ) AS steamid_rank,
   @current_steamid := steamid
   FROM Games_1
   WHERE playtime_forever > 120
   ORDER BY steamid, playtime_forever DESC
 ) ranked
 WHERE steamid_rank <= 5;

# piped to tsv (avoids file system permission errors)
mysql -u user --password=pw --database=steamdb --execute='SELECT steamid, appid, playtime_forever FROM ( SELECT steamid, appid, playtime_forever, @steamid_rank := IF(@current_steamid = steamid, @steamid_rank + 1, 1) AS steamid_rank, @current_steamid := steamid FROM Games_1 WHERE playtime_forever > 120 ORDER BY steamid, playtime_forever DESC) ranked WHERE steamid_rank <= 5' -q -n -B -r > games_1.tsv

# convert tsv to csv
mv games_1.tsv games_1.csv
sed -i '/\t/ s//,/g' games_1.csv

# get titles
mysql -u user --password=pw --database=steamdb --execute='SELECT appid, Title FROM App_ID_Info WHERE Type = "game";' -q -n -B -r > app_title.csv && sed -i '/\t/ s//,/g' app_title.csv

# get genres
mysql -u user --password=pw --database=steamdb --execute='SELECT appid, Genre FROM Games_Genres;' -q -n -B -r > app_genres.csv && sed -i '/\t/ s//,/g' app_genres.csv
```
