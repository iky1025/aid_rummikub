from rummikub_solver import parse_tiles
from rummikub_env import RummikubEnv


def input_table_sets():
    table_sets = []

    print("\n테이블 세트를 한 줄에 하나씩 입력하세요.")
    print("예: R3 R4 R5")
    print("테이블이 비어 있으면 그냥 Enter를 누르세요.")
    print("입력을 끝내려면 빈 줄을 누르세요.")

    while True:
        line = input("테이블 세트 입력: ").strip()

        if line == "":
            break

        table_sets.append(parse_tiles(line))

    return table_sets


def print_candidate_results(results):
    print("\n=== 여러 ILP 후보 ===")

    if not results:
        print("후보 없음")
        return

    for i, result in enumerate(results):
        print(f"\n--- 후보 {i} ---")
        print("새로 사용한 손패 타일 수:", result.used_hand_tile_count)
        print("새로 사용한 손패 점수:", result.used_hand_score)

        print("선택된 세트:")
        for j, tile_set in enumerate(result.selected_sets):
            print(f"  {j}: {tile_set}")

        if result.remaining_hand:
            from rummikub_solver import format_tiles
            print("남은 손패:", format_tiles(result.remaining_hand))
        else:
            print("남은 손패: 없음")


def main():
    print("=== 루미큐브 환경 + ILP Solver ===")
    print("게임 시작 시 랜덤으로 손패 14장을 받습니다.")
    print("타일 표기 예: R1 B13 Y7 K10 J")
    print("색상: R, B, Y, K / 조커: J")

    table_sets = input_table_sets()

    env = RummikubEnv(hand_size=14)

    env.reset(
        table_sets=table_sets,
        shuffle=True,
    )

    print("\n초기 상태")
    env.render()

    results = env.solve_candidate_moves(max_candidates=10)

    print_candidate_results(results)

    if results:
        answer = input("\n몇 번 후보를 적용할까요? 적용하지 않으려면 Enter: ").strip()

        if answer != "":
            index = int(answer)

            if 0 <= index < len(results):
                env.apply_solution(results[index])

                print("\n적용 후 상태")
                env.render()
            else:
                print("잘못된 후보 번호입니다.")
        else:
            print("아무 후보도 적용하지 않았습니다.")
    else:
        print("\n낼 수 있는 손패 조합이 없습니다.")


if __name__ == "__main__":
    main()