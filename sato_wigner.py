import json
import os
from dataclasses import dataclass
from functools import reduce

import numpy as np
from tqdm import tqdm

OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass(frozen=True)
class ResultKey:
    n_qubit: int
    seed: int

    @classmethod
    def from_filename(cls, filename: str) -> "ResultKey":
        """
        ファイル名 {n_qubit}_{seed}.json から n_qubit, seed を抽出して ResultKey を生成する
        """
        basename = os.path.basename(filename)
        if basename.endswith(".json"):
            basename = basename[:-5]
        try:
            n_qubit_str, seed_str = basename.split("_")
            n_qubit = int(n_qubit_str)
            seed = int(seed_str)
        except Exception as e:
            raise ValueError(
                f"ファイル名 '{filename}' から n_qubit, seed を抽出できません: {e}")
        obj = cls(n_qubit=n_qubit, seed=seed)
        return obj

    def to_filename(self) -> str:
        return f"{self.n_qubit}_{self.seed}.json"


class KeySet:
    def __init__(self, keys: list[ResultKey]) -> None:
        self._keys = keys

    @classmethod
    def from_folder(cls, folder: str) -> "KeySet":
        return cls([ResultKey.from_filename(f) for f in os.listdir(folder) if f.endswith(".json")])

    def has_key(self, n_qubit: int, seed: int) -> bool:
        for key in self._keys:
            if key.n_qubit == n_qubit and key.seed == seed:
                return True
        return False


def main_worker(nqubit, seed):
    rng = np.random.RandomState(seed)

    # 基本ゲート
    from qulacs.gate import X, Z
    I_mat = np.eye(2, dtype=complex)
    X_mat = X(0).get_matrix()
    Z_mat = Z(0).get_matrix()
    Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)

    def make_fullgate(list_SiteAndOperator, nqubit):
        '''
        list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
        関係ないqubitにIdentityを挿入して
        I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
        という(2**nqubit, 2**nqubit)行列をつくる.
        '''
        list_Site = [SiteAndOperator[0]
                     for SiteAndOperator in list_SiteAndOperator]
        list_SingleGates = []  # 1-qubit gateを並べてnp.kronでreduceする
        cnt = 0
        for i in range(nqubit):
            if (i in list_Site):
                list_SingleGates.append(list_SiteAndOperator[cnt][1])
                cnt += 1
            else:  # 何もないsiteはidentity
                list_SingleGates.append(I_mat)

        return reduce(np.kron, list_SingleGates)

    Bx_list = rng.uniform(-1, 1, nqubit)
    By_list = rng.uniform(-1, 1, nqubit)
    Bz_list = rng.uniform(-1, 1, nqubit)
    Jmat = np.zeros((nqubit, nqubit), dtype=complex)
    for i in range(nqubit):
        for j in range(nqubit):
            Jmat[i][j] = rng.uniform(-3, 3)

    #####################
    sca_z = 0.73
    #####################

    dataham = np.zeros((2 ** nqubit, 2 ** nqubit), dtype=complex)

    # ハミルトニアンを作成

    for i in range(nqubit):
        B = Bx_list[i]
        dataham += B * make_fullgate([[i, X_mat]], nqubit)
        B = By_list[i]
        dataham += B * make_fullgate([[i, Y_mat]], nqubit)
        B = Bz_list[i]
        dataham += B * make_fullgate([[i, Z_mat]], nqubit)

        for j in range(nqubit):
            dataham += Jmat[i][j] * \
                       make_fullgate([[i, X_mat], [j, X_mat]], nqubit)

            dataham += Jmat[i][j] * \
                       make_fullgate([[i, Y_mat], [j, Y_mat]], nqubit)

            dataham += sca_z * Jmat[i][j] * \
                       make_fullgate([[i, Z_mat], [j, Z_mat]], nqubit)

    diag, eigen_vecs = np.linalg.eigh(dataham)

    # --- ここからがJSONへの書き出し処理 ---

    # 1. 保存するデータを含む辞書を作成
    #    JSONはNumpy配列を直接扱えないため、.tolist()で通常のリストに変換する
    output_data = {
        "nqubit": nqubit,
        "seed": seed,
        "diag": diag.tolist()
    }

    # 3. 出力ファイル名を生成
    filename = ResultKey(n_qubit=nqubit, seed=seed).to_filename()
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 4. JSONファイルにデータを書き込む
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=4)


def list_result_keys(output_dir: str) -> list[ResultKey]:
    return [ResultKey.from_filename(f) for f in os.listdir(output_dir) if f.endswith(".json")]


def run_with_param(target_n_qubit: int, target_seed_list: list[int], cpu_count: int):
    print("run_with_param")
    print(" - target_n_qubit:", target_n_qubit)
    print(" - target_seed_list:", target_seed_list)
    print(" - cpu_count:", cpu_count)

    key_set = KeySet.from_folder(OUTPUT_DIR)

    # 足りないキーを生成
    insufficient_keys: list[ResultKey] = []
    for target_seed in target_seed_list:
        if not key_set.has_key(target_n_qubit, target_seed):
            insufficient_keys.append(
                ResultKey(n_qubit=target_n_qubit, seed=target_seed)
            )

    # joblibを使って並列実行
    from joblib import Parallel, delayed
    Parallel(n_jobs=cpu_count)(
        delayed(main_worker)(key.n_qubit, key.seed) for key in tqdm(insufficient_keys)
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "指定した量子ビット数と実行回数でWigner計算を実行し、"
            "結果をJSONファイルとして保存します。\n"
            "n_qubit: 量子ビット数\n"
            "run_count: 実行回数（seed=1からrun_countまで）\n"
            "cpu_count: 実行に使用するCPU数"
        )
    )
    parser.add_argument(
        "n_qubit",
        type=int,
        help=(
            "量子ビット数を指定します。\n"
            "例: 10"
        )
    )
    parser.add_argument(
        "run_count",
        type=int,
        help=(
            "実行回数（seed=1からrun_countまで）を指定します。\n"
            "例: 3 なら seed=1,2,3 で実行"
        )
    )
    max_cpu_count = os.cpu_count() - 1
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=max_cpu_count,
        help=(
            "実行に使用するCPU数を指定します。指定なしの場合、最大のCPU数が使用されます。最大のCPU数はマシンに実装されている(コア数-1)です。"
        )
    )
    args = parser.parse_args()
    n_qubit = args.n_qubit
    run_count = args.run_count
    cpu_count = args.cpu_count
    if cpu_count > max_cpu_count:
        cpu_count = max_cpu_count
    target_seed_list = list(range(1, run_count + 1))
    run_with_param(target_n_qubit=n_qubit, target_seed_list=target_seed_list, cpu_count=cpu_count)


# main関数を呼び出す
if __name__ == "__main__":
    main()
