def draw_wmma_instr_cmd(waveSize):
    wmma_mode = 0 if waveSize == 32 else 1
    return f'''\\begin{{document}}
  \\begin{{tikzpicture}}
    \\def\\scale{{1}}
    \\coordinate (C TL) at (0,0);
    \\def\\elem{{0.25}}
    \\drawWMMAInstr{{{wmma_mode}}}{{1}}
  \\end{{tikzpicture}}
\\end{{document}}'''


def generate_wmma_tex(args):
    assert args.plot_type == "wmma", \
        f"parsing the wrong arguments. Want wmma but have {args.plot_type}"
    # preprocess the args
    waveSize = args.wave_size
    # checks and logging
    # write the tex file
    with open("myplot.tex", 'w') as f_plot:
        with open("utils/preamble.tex") as file:
            preamble = file.read()

        f_plot.write(preamble)
        draw_wmma_str = draw_wmma_instr_cmd(waveSize)
        f_plot.write("\input{wmma/wmmaLayout}\n")
        f_plot.write(draw_wmma_str)
