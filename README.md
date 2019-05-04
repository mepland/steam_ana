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

## Downloading Data
Download `steam.sql.gz` from [steam.internet.byu.edu](http://steam.internet.byu.edu/), then extract with gunzip. Note that the download size is 18GB and the extracted size is 161GB so you'll probably need a decent workstation. Then load into MySQL:  

```bash
# download and extract
wget http://steam.phoenixteam.net/steam.sql.gz
gunzip steam.sql.gz

# load into mysql
mysql -u root -p
CREATE DATABASE steamdb; USE steamdb;
SET autocommit=0; source steam.sql; COMMIT;

# extract useful data

# https://stackoverflow.com/questions/5859391/create-a-temporary-table-in-a-select-statement-without-a-separate-create-table
CREATE TEMPORARY TABLE IF NOT EXISTS twohours

# first look
SELECT steamid, appid, playtime_forever FROM GAMES_1 WHERE playtime_forever > 120 LIMIT 50;

# save out values to create graph
# https://www.databasejournal.com/features/mysql/selecting-the-top-n-results-by-group-in-mysql.html
# https://stackoverflow.com/questions/356578/how-to-output-mysql-query-results-in-csv-format/356605#356605

SELECT steamid, appid, playtime_forever
 FROM
 (
   SELECT steamid, appid, playtime_forever,
   @steamid_rank := IF(@current_steamid = steamid,
                         @steamid_rank + 1,
                         1
                      ) AS steamid_rank,
   @current_steamid := steamid
   FROM GAMES_1
   WHERE playtime_forever > 120
   ORDER BY steamid, playtime_forever DESC
 ) ranked
 WHERE steamid_rank <= 5
INTO OUTFILE 'games1.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n';
```
