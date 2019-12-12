# Concept Codes from the UW Challenge Data
The list of codes in these files represent the OMOP concept ids that are present in more than 100 patients in the UW Challenge Data. We list each table individually as well as combine all the codes into one table (all_concepts.csv).
## Concepts
| Table | Concept Count | Vocabulary |
|-------|-----------------|------------|
|Condtion Occurrence | 5742 | SNOMED |
|Drug Exposure | 2474 | RxNorm |
|Observation | 333 | SNOMED, HCPCS |
|Measurement | 571 | SNOMED, LOINC |
|Procedure Occurrence | 2247 | SNOMED, CPT4|
|Person | 6 | Race |
|Visit Occurrence | 3 | Visit |
|All concepts | 11376 |  |
## Notes
One caviat to keep in mind when using these codes.
- Procedure Occurrence has both SNOMED and CPT4 codes. There are some redundancies that you'll want to account for when using procedures. An example of this is the codes 2110195 (CPT4) and 4238715 (SNOMED) both of which refer to the procedure *Insertion of intrauterine device (IUD)*.