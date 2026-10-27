"""Star and profile MSA on synthetic reads."""

from gen import rand_seq

from psnw_py import align, msa
from psnw_py.core import BAM_CDEL, BAM_CINS, BAM_CMATCH


def _check_rows(m, ref, ref_offset, seqs):
    """REF row equals ref at every reference column; each read row, ungapped,
    equals its (clip-stripped) sequence."""
    L = m.ncols
    assert all(len(r) == L for r in m.rows)
    ref_row = m.rows[0]
    for c in range(L):
        rp = int(m.ref_pos[c])
        if rp >= 0:
            assert ref_row[c] == ref[rp - ref_offset]
            assert m.anchor[c] == rp
        else:
            assert ref_row[c] == "-"
    # reference columns appear in order, each once
    rps = [int(x) for x in m.ref_pos if x >= 0]
    assert rps == sorted(rps) and len(set(rps)) == len(rps)
    for name, seq in seqs.items():
        assert m.row(name).replace("-", "") == seq


def test_star_columns_map_to_reference():
    ref = "ACGTACGTAAGGCCTTACGTGCAT"
    off = 1000
    reads = [
        ("wt", ref[2:20], off + 2, "18M"),
        ("del", ref[0:8] + ref[11:24], off + 0, [(BAM_CMATCH, 8), (BAM_CDEL, 3), (BAM_CMATCH, 13)]),
        ("ins2", ref[4:10] + "TT" + ref[10:18], off + 4, "6M2I8M"),
        ("ins3", "GG" + ref[6:10] + "CAT" + ref[10:16], off + 6, "2S4M3I6M"),
    ]
    m = msa.star(ref, reads, ref_offset=off)
    _check_rows(
        m, ref, off,
        {"wt": ref[2:20], "del": ref[0:8] + ref[11:24], "ins2": ref[4:10] + "TT" + ref[10:18],
         "ins3": ref[6:10] + "CAT" + ref[10:16]},
    )
    ins_cols = m.insertion_columns(off + 10)
    assert len(ins_cols) == 3 and m.insertion_columns() == ins_cols
    assert "".join(m.row("ins2")[c] for c in ins_cols) == "TT-"
    assert "".join(m.row("ins3")[c] for c in ins_cols) == "CAT"
    assert "".join(m.row("wt")[c] for c in ins_cols) == "---"
    c = m.column_of(off + 9)
    assert m.row("del")[c] == "-" and m.row("wt")[c] == ref[9]
    assert m.rows[0].count("-") == 3
    fa = m.to_fasta()
    assert fa.startswith(">REF\n") and fa.count(">") == 5


def test_star_is_order_independent():
    ref = "ACGTACGTAAGGCCTTACGTGCAT"
    reads = [("a", ref[0:10] + "GG" + ref[10:20], 0, "10M2I10M"), ("b", ref[5:24], 5, "19M")]
    m1 = msa.star(ref, reads)
    m2 = msa.star(ref, reads[::-1])
    assert m1.row("a") == m2.row("a") and m1.row("b") == m2.row("b")
    assert list(m1.ref_pos) == list(m2.ref_pos)


def test_star_from_pairwise_placements(rng):
    ref = rand_seq(rng, 300)
    reads = []
    for k in range(10):
        s = rng.randint(0, 50)
        seq = ref[s:150] + ("ACT" if k % 3 == 0 else "") + ref[150 + (5 if k % 3 == 1 else 0):260]
        aln = align(ref, seq, free_target_ends=True)
        reads.append((f"r{k}", seq, aln.target_start, list(aln.cigartuples)))
    m = msa.star(ref, reads)
    _check_rows(m, ref, 0, {r[0]: r[1] for r in reads})


def test_profile_aligns_shared_insertion(rng):
    ref = rand_seq(rng, 160)
    ins = "GATTACAGG"
    cut = 80
    reads = []
    for k in range(10):
        s = rng.randint(0, 40)
        e = rng.randint(120, 160)
        has = k % 2 == 0
        seq = ref[s:cut] + (ins if has else "") + ref[cut:e]
        if k == 4:  # a mismatch next to the insertion must not split the columns
            seq = seq[: cut - s - 2] + ("A" if seq[cut - s - 2] != "A" else "C") + seq[cut - s - 1 :]
        reads.append((f"r{k}", seq, has))
    m = msa.profile(ref, [(n, s) for n, s, _ in reads])
    _check_rows(m, ref, 0, {n: s for n, s, _ in reads})
    ins_cols = m.insertion_columns()
    assert len(ins_cols) == len(ins)
    for name, seq, has in reads:
        got = "".join(m.row(name)[c] for c in ins_cols)
        assert got == (ins if has else "-" * len(ins)), name
        pos0, cig = m.placements[name]
        if has:
            assert (BAM_CINS, len(ins)) in cig
        else:
            assert all(op == BAM_CMATCH for op, _ in cig)


def test_profile_placement_reconstructs_read(rng):
    ref = rand_seq(rng, 200)
    reads = []
    for k in range(12):
        s = rng.randint(0, 30)
        e = rng.randint(150, 200)
        seq = ref[s:90] + ("TTG" if k % 3 == 0 else "") + ref[90 + (7 if k % 3 == 1 else 0):e]
        reads.append((f"r{k}", seq))
    m = msa.profile(ref, reads, pse=[5 if 85 <= i < 105 else 0 for i in range(200)], order="length")
    for name, seq in reads:
        pos0, cig = m.placements[name]
        r, y = pos0, 0
        for op, n in cig:
            if op == BAM_CMATCH:
                assert seq[y : y + n] == ref[r : r + n]
                r += n
                y += n
            elif op == BAM_CINS:
                y += n
            elif op == BAM_CDEL:
                r += n
        assert y == len(seq)
    assert m.order == [n for n, _ in sorted(reads, key=lambda x: (-len(x[1]), x[0]))]
    assert m.names == ["REF"] + [n for n, _ in reads]


def test_profile_single_read_matches_pairwise(rng):
    for _ in range(20):
        ref = rand_seq(rng, 80)
        seq = ref[:30] + ref[37:70] + "AC" + ref[70:]
        pse = [5 if 25 <= i < 45 else 0 for i in range(80)]
        pw = align(ref, seq, pse=pse)
        pm = msa.profile(ref, [("r", seq)], pse=pse, free_ends=False)
        assert pm.scores["r"] == pw.score
        assert pm.placements["r"] == (0, list(pw.cigartuples))


def test_profile_fill_char(rng):
    ref = rand_seq(rng, 60)
    m = msa.profile(ref, [("a", ref[10:50]), ("b", ref[0:30] + "GGG" + ref[30:60])], fill=".")
    a = m.row("a")
    assert a.startswith("." * 10) and a.endswith(".")
    ins = m.insertion_columns()
    assert len(ins) == 3 and all(m.row("a")[c] == "-" for c in ins)
