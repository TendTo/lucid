import { useMemo, useSyncExternalStore } from "react";

function subscribe(callback: () => void) {
  window.addEventListener("resize", callback);
  return () => {
    window.removeEventListener("resize", callback);
  };
}

export function useDimensions(ref: React.RefObject<HTMLElement | null>): {
  width: number;
  height: number;
} {
  const dimensions = useSyncExternalStore(subscribe, () =>
    JSON.stringify({
      width: ref.current?.offsetWidth ?? 0, // 0 is default width
      height: ref.current?.offsetHeight ?? 0, // 0 is default height
    })
  );
  return useMemo(() => JSON.parse(dimensions), [dimensions]);
}
