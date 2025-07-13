import { useLayoutEffect, useRef } from "react";

type DangerousElementProps = {
  markup: string;
};

export function DangerousElement({ markup }: DangerousElementProps) {
  const elRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (!elRef.current) return;
    
    elRef.current.replaceChildren();
    
    const range = document.createRange();
    range.selectNode(elRef.current);
    const documentFragment = range.createContextualFragment(markup);

    // Inject the markup, triggering a re-run!
    elRef.current.innerHTML = "";
    elRef.current.append(documentFragment);
  }, [markup]);

  return <div className="dangerous-element" ref={elRef} />;
}
