from domain_specific_machine_translation.ii_sentence_mining.parallel_sentence_miner import ParallelSentenceMiner


if __name__ == '__main__':
    miner = ParallelSentenceMiner(src_lang="en", trg_lang="de")
    for i in range(1, 8):
        print(f"\nFile {i}: ")
        miner.mine_and_add_pairs(file_src=f"../i_raw_text_files/hp_{i}_en.txt",
                                 file_trg=f"../i_raw_text_files/hp_{i}_de.txt")
    miner.write_output_txt_files()
