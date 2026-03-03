use lac_rs::{Lac, Mode};

fn main() {
    let lac = Lac::new(Mode::Lac).expect("load lac model");
    let seg = Lac::new(Mode::Seg).expect("load seg model");
    for text in [
        "百度是一家高科技公司",
        "LAC是个优秀的分词工具",
        "春天的花开秋天的风以及冬天的落阳",
    ] {
        let result = lac.run(text);
        let formatted: Vec<String> = result
            .iter()
            .map(|item| format!("{}/{}", item.word, item.tag))
            .collect();
        let seg_words = seg.cut(text);
        println!(
            "{text}\nLAC  {}\nSEG  {}\n",
            formatted.join(" "),
            seg_words.join(" ")
        );
    }
}
