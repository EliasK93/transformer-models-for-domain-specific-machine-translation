from domain_specific_machine_translation import model_utils
from domain_specific_machine_translation.file_utils import read_txt, write_txt


if __name__ == '__main__':

    # sample sentence pairs (source sentences, reference translations)
    src_text_list = read_txt("../iii_model_training/training_data/txt/test_src.txt")
    trg_text_list = read_txt("../iii_model_training/training_data/txt/test_trg.txt")

    # write source sentences to file
    write_txt(src_text_list, "txt/sources.txt")

    # write reference translations to file
    write_txt(trg_text_list, "txt/targets_reference.txt")

    # load pretrained (non-finetuned) models, generate translations and write to file
    for model_type in ["Helsinki-NLP/opus-mt-en-de", "t5-large", "facebook/nllb-200-distilled-600M"]:
        for fine_tuned in [False, True]:
            tokenizer, model = model_utils.load_model(model_type, local=fine_tuned)
            results = model_utils.translate(src_text_list, model_type, model, tokenizer)
            out_file_name = "txt/targets_" + model_type.split("/")[-1] + ("_finetuned" if fine_tuned else "") + ".txt"
            write_txt(results, out_file_name)
