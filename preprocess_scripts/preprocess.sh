lang=es

ROOT=/data/zhangshaolei/StreamSpeech
PREPROCESS_ROOT=$ROOT/preprocess_scripts

bash $PREPROCESS_ROOT/1.learn_KM_clustering_model.sh $lang
echo 'finish 1.learn_KM_clustering_model.sh'

bash $PREPROCESS_ROOT/2.prep_cvss_c_multilingual_data.sh $lang
echo 'finish 2.prep_cvss_c_multilingual_data.sh'

bash $PREPROCESS_ROOT/3.prep_cvss_c_multitask_data.sh $lang
echo 'finish 3.prep_cvss_c_multitask_data.sh'

bash $PREPROCESS_ROOT/5.prep_cvss_c_ref_txt.sh $lang
echo 'finish 5.prep_cvss_c_ref_txt.sh'

bash $PREPROCESS_ROOT/6.extract_simuleval_data.sh $lang
echo 'finish 6.extract_simuleval_data.sh'

bash $PREPROCESS_ROOT/7.prep_cvss_c_multitask_asr_data.sh $lang
echo 'finish 7.prep_cvss_c_multitask_asr_data.sh'

bash $PREPROCESS_ROOT/8.prep_cvss_c_simuleval_unit.sh $lang
bash $PREPROCESS_ROOT/8.prep_cvss_c_simuleval_src.sh $lang
echo 'finish 8.prep_cvss_c_simuleval_unit.sh, 8.prep_cvss_c_simuleval_src.sh '

# # only for s2tt training on CVSS-C
# bash $PREPROCESS_ROOT/9.prep_cvss_c_s2st_mtl_data.sh  $lang
# echo 'finish 9.prep_cvss_c_s2st_mtl_data.sh'