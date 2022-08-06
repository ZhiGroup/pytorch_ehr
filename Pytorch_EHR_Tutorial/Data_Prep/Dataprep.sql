/*Explore Data*/
select * from patients
where id='af51fb48-22f9-4ffb-a843-8b840afc6976'


select * from encounters
where patient='af51fb48-22f9-4ffb-a843-8b840afc6976'


select * from conditions
where patient='af51fb48-22f9-4ffb-a843-8b840afc6976'

select * from conditions
where lower(DESCRIPTION) like '%diabet%'

select * from conditions
where lower(DESCRIPTION) like '%heart%' --- 84114007,88805009

---Baseline Cohort
select distinct patient from conditions where code in ('44054006','15777000')---41,868 baseline

--Cases
select distinct patient from conditions
where patient in (select distinct patient from conditions where code in ('44054006','15777000'))
and  code in ('84114007','88805009')--- 4590

---if it hf before or in the same day of D will exclude, otherwise
select distinct a.patient, first_db_diag,first_hf_diag  from
(select patient, max(first_dbp_diag) first_db_diag -------- logically patients will be first diagenosed with diabetes after being first diagnosed with prediabetes, therefore, I give priority to first diag of confirmed diabetes, but this synthetic :)
from (select patient,code, min(start) first_dbp_diag from conditions where code in ('44054006','15777000') group by patient,code)x
group by patient)a 
left join
(select patient,min(start) first_hf_diag from conditions
where patient in (select distinct patient from conditions where code in ('44054006','15777000')) and  code in ('84114007','88805009') group by patient)b
on a.patient=b.patient
where first_hf_diag <= first_db_diag


/* patients to exclude
009b3b30-c8db-4f23-81ab-e5c1ab3b3e3e
066e1132-0153-4821-9077-4f881c3b531c
12980423-b974-49f0-96fc-07a7e971c9ac
3bc39b1b-d40e-4d5d-8c0b-7c635de123be
7ca7f41c-4cf2-4903-91ba-5ed299ec2faf
cd4acd58-c859-42ca-a19e-4c28afb1aa64*/




---Create the control patient list along with the index date ( for this example, we will predict the patient risk for HF on their first diabetes or prediabetes diagnosis
--- for the tutorial purpose with computational recsource constraint on colab, I will only use diabetic patients control, that would recduce my controls to 7,545 instead of 37,278 ( in real scenario I will keep them)
create table dhf_control_v1 as
select distinct a.patient, first_db_diag,last_db_diag  from
(select patient, max(first_dbp_diag) first_db_diag, max(last_db_diag) last_db_diag
from (select patient,code, min(start) first_dbp_diag, max(start) last_db_diag from conditions where code in ('44054006') group by patient,code)x
group by patient)a 
left join
(select patient,min(start) first_hf_diag from conditions
where patient in (select distinct patient from conditions where code in ('44054006','15777000')) and  code in ('84114007','88805009') group by patient)b
on a.patient=b.patient
where first_hf_diag is null

--- create cases, and in order to make the problem harder, but more useful, we will restrict cases to whom develop HF at least 90 days after but will keep patients with prediabetes to have 4581 cases 
create table dhf_cases_v1 as
select distinct a.patient, first_db_diag,first_hf_diag  from
(select patient, max(first_dbp_diag) first_db_diag
from (select patient,code, min(start) first_dbp_diag from conditions where code in ('44054006','15777000') group by patient,code)x
group by patient)a 
left join
(select patient,min(start) first_hf_diag from conditions
where patient in (select distinct patient from conditions where code in ('44054006','15777000')) and  code in ('84114007','88805009') group by patient)b
on a.patient=b.patient
where first_hf_diag is not NULL
and date(first_hf_diag )> date(first_db_diag, '+90 days')


select * from conditions where patient='001b34da-747c-4526-a83d-b2b9494e4d6b' ----1972-03-24	2020-01-08
order by start desc

----- so now, we have our cohort, and can create the label file, and we can further clean

select patient, 1 label, *, (Julianday(first_hf_diag) - julianday(first_db_diag)) tte
from dhf_cases_v1 

select patient, 0 label, *, (Julianday(last_db_diag) - julianday(first_db_diag)) tte ----- oh last db date is commonly the same as the first here ( we used db only not pre --- so let's use the cuttof as the last encounter discharege date
from dhf_control_v1 

create table dhf_control_v2 as
select y.*,x.last_enc_ad from
dhf_control_v1 y
left join (select patient, date(max(start)) last_enc_ad from encounters
where patient in (select patient from dhf_control_v1)
group by patient)x
on x.patient= y.patient
---where last_enc_ad is null ---0 rows perfect

create table dhf_label_v1
as
select patient, 1 label, (Julianday(first_hf_diag) - julianday(first_db_diag)) tte , first_db_diag, first_hf_diag cuttoff
from dhf_cases_v1 
union 
select patient, 0 label, (Julianday(last_enc_ad) - julianday(first_db_diag)) tte ----- oh last db date is commonly the same as the first here ( we used db only not pre --- so let's use the cuttof as the last encounter discharege date
, first_db_diag, last_enc_ad cuttoff
from dhf_control_v2

select * from dhf_label_v1
where tte is null --- 0 good

-----Let's create the data

select * from 
(select distinct patient,id encounter, date(start) adm_dt, date(stop) disc_dt
from encounters
where patient in (select patient from dhf_label_v1)) b
where patient is null
or ifnull(adm_dt, disc_dt) is null ---0 rows, perfect


create TABLE dhf_hard_encs_v1 AS
select b.* 
from dhf_label_v1 a
left join 
(select distinct patient,id encounter, date(start) adm_dt, date(stop) disc_dt
from encounters
where patient in (select patient from dhf_label_v1)) b
on a.patient=b.patient
and ifnull(adm_dt, disc_dt)<=first_db_diag

select * from dhf_hard_encs_v1
where patient isnull ---- 344 rows (i.e some patient in condition not have encounters ?? -- data quality issue, better drop those patients

drop TABLE dhf_hard_encs_v1

create TABLE dhf_hard_encs_v1 AS
select b.* 
from dhf_label_v1 a
inner join  ----> use inner join
(select distinct patient,id encounter, date(start) adm_dt, date(stop) disc_dt
from encounters
where patient in (select patient from dhf_label_v1)) b
on a.patient=b.patient
and ifnull(adm_dt, disc_dt)<=first_db_diag

select * from dhf_hard_encs_v1
where patient isnull ---- 0 perfect

create TABLE dhf_easy_encs_v1 AS
select  b.*
from dhf_label_v1 a
inner join 
(select distinct patient,id encounter, date(start) adm_dt, date(stop) disc_dt
from encounters
where patient in (select patient from dhf_label_v1)) b
on a.patient=b.patient
and ifnull(adm_dt, disc_dt)<= date(cuttoff, '-90 days')

create TABLE dhf_easy_label_v1 AS 
select x.*, (julianday(cuttoff)-Julianday(new_index)) new_tte from
(select  distinct a.* ,date(cuttoff, '-90 days') new_preindex, max (ifnull(adm_dt, disc_dt)) over (PARTITION by b.patient) new_index
from dhf_label_v1 a
inner join 
(select distinct patient,id encounter, date(start) adm_dt, date(stop) disc_dt
from encounters
where patient in (select patient from dhf_label_v1)) b
on a.patient=b.patient
and ifnull(adm_dt, disc_dt)<= date(cuttoff, '-90 days'))x

-----We will extract the data based on the encounters

select encounter, count(distinct patient) pcnt from dhf_easy_encs_v1
group by ENCOUNTER
order by pcnt DESC ---- good ( I have a peace of mind that all encounters are 1:1 for pts)

---diagnosis
select distinct a.patient, 'D_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a  
inner join conditions b
on a.patient=b.patient and a.encounter=b.encounter


select distinct a.patient, 'M_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a 
inner join medications b
on a.patient=b.patient and a.encounter=b.encounter
where ifnull(b.DISPENSES,0)>0

-----similarly you can select procedures,obervations ...etc

create table  dhf_easy_data_v1
as 
select distinct a.patient, 'D_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a  
inner join conditions b
on a.patient=b.patient and a.encounter=b.encounter
union
select distinct a.patient, 'M_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a 
inner join medications b
on a.patient=b.patient and a.encounter=b.encounter
where ifnull(b.DISPENSES,0)>0


create table  dhf_easy_data_v1
as 
select distinct a.patient, 'D_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a  
inner join conditions b
on a.patient=b.patient and a.encounter=b.encounter
union
select distinct a.patient, 'M_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a 
inner join medications b
on a.patient=b.patient and a.encounter=b.encounter
where ifnull(b.DISPENSES,0)>0

---- One way to add demographics is to add the information for each patient at each visit for example
select *, cast(strftime('%Y.%m%d', disc_dt) - strftime('%Y.%m%d', BIRTHDATE ) as int)age from ---- This table can also be used for the discriptive analysis.
(select distinct patient, disc_dt from dhf_easy_data_v1) a
left join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
--where race is NULL or ETHNICITY is null or gender is null or birthdate is null ---- 0 rows, perfect

---- then we can do as FOLLOWING

create table dhf_easy_data_v1_dmd as
select * from dhf_easy_data_v1
union
select distinct a.patient, 'a_'||cast(strftime('%Y.%m%d', disc_dt) - strftime('%Y.%m%d', BIRTHDATE ) as int) ecode, disc_dt from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'r_'||lower(race)  ecode, disc_dt from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'r_'||lower(ETHNICITY)  ecode, disc_dt  from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'g_'||lower(gender)  ecode, disc_dt  from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient



create table  dhf_easy_data_v1
as 
select distinct a.patient, 'D_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a  
inner join conditions b
on a.patient=b.patient and a.encounter=b.encounter
union
select distinct a.patient, 'M_'||b.code ecode, disc_dt
from dhf_easy_encs_v1 a 
inner join medications b
on a.patient=b.patient and a.encounter=b.encounter
where ifnull(b.DISPENSES,0)>0

---- One way to add demographics is to add the information for each patient at each visit for example
select *, cast(strftime('%Y.%m%d', disc_dt) - strftime('%Y.%m%d', BIRTHDATE ) as int)age from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
left join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
--where race is NULL or ETHNICITY is null or gender is null or birthdate is null ---- 0 rows, perfect

---- then we can do as FOLLOWING

create table dhf_easy_data_v1_dmd as
select * from dhf_easy_data_v1
union
select distinct a.patient, 'a_'||cast(strftime('%Y.%m%d', disc_dt) - strftime('%Y.%m%d', BIRTHDATE ) as int) ecode, disc_dt from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'r_'||lower(race)  ecode, disc_dt from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'r_'||lower(ETHNICITY)  ecode, disc_dt  from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'g_'||lower(gender)  ecode, disc_dt  from
(select distinct patient, disc_dt from dhf_easy_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient

select * from dhf_easy_data_v1_dmd

drop table dhf_easy_data_v1 ---- keep my db clean :) 


/* Redo the same for the hard prediction cohort*/


create table  dhf_hard_data_v1
as 
select distinct a.patient, 'D_'||b.code ecode, disc_dt
from dhf_hard_encs_v1 a  
inner join conditions b
on a.patient=b.patient and a.encounter=b.encounter
union
select distinct a.patient, 'M_'||b.code ecode, disc_dt
from dhf_hard_encs_v1 a 
inner join medications b
on a.patient=b.patient and a.encounter=b.encounter
where ifnull(b.DISPENSES,0)>0


create table dhf_hard_data_v1_dmd as
select * from dhf_hard_data_v1
union
select distinct a.patient, 'a_'||cast(strftime('%Y.%m%d', disc_dt) - strftime('%Y.%m%d', BIRTHDATE ) as int) ecode, disc_dt from
(select distinct patient, disc_dt from dhf_hard_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'r_'||lower(race)  ecode, disc_dt from
(select distinct patient, disc_dt from dhf_hard_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'r_'||lower(ETHNICITY)  ecode, disc_dt  from
(select distinct patient, disc_dt from dhf_hard_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient
UNION
select distinct a.patient, 'g_'||lower(gender)  ecode, disc_dt  from
(select distinct patient, disc_dt from dhf_hard_data_v1) a
inner join(select distinct id patient, race,ethnicity,gender, BIRTHDATE from patients) b
on a.patient=b.patient


 

drop table dhf_hard_data_v1


---+++ Now my data is extracted, last thing to check 

select count (distinct patient) from  dhf_hard_data_v1_dmd ----11740

select count (distinct patient) from dhf_label_v1 ----12126 ---- I need to restrict my label here to only those hard patients

create table dhf_hard_label_v1 as
select distinct patient, label, tte from dhf_label_v1
where patient in (select distinct patient from  dhf_hard_data_v1_dmd)

---check the easy

select count (distinct patient) from  dhf_easy_data_v1_dmd ----12,119 ( why more :) )

select count (distinct patient) from dhf_easy_label_v1 ----12119 (good)

create table dhf_hard_label_v1 as
select distinct patient, label, tte from dhf_label_v1
where patient in (select distinct patient from  dhf_hard_data_v1_dmd)

---when I extract the data I limit to 
select distinct patient, label, new_tte from dhf_easy_label_v1

----so now I can extract the dhf_hard_data_v1_dmd,dhf_hard_label_v1 , dhf_easy_data_v1_dmd, and the query above
create table dhf_easy_label_v1_e as
select distinct patient, label, new_tte from dhf_easy_label_v1 ---- I have to create a table to export it

----- will need trhe description for explanation :)
create table dhf_codes_desc as
select distinct 'D_'||code ecode, 'Diag' , DESCRIPTION from conditions 
union 
select distinct 'M_'||code ecode, 'Med' , DESCRIPTION from medications
