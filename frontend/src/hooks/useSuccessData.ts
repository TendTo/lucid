import { useReducer, useCallback } from "react";
import type { SuccessResponseData } from "@/types/types";
import type { PlotParams } from "react-plotly.js";
import { emptyFigure } from "@/utils/constants";

type SuccessDataState = SuccessResponseData;

type SuccessDataAction =
  | { type: "SET_SUCCESS"; payload: SuccessResponseData }
  | {
      type: "UPDATE_FIELD";
      field: keyof SuccessResponseData;
      value: SuccessResponseData[keyof SuccessResponseData];
    }
  | { type: "UPDATE_FIGURE"; value: string | PlotParams }
  | { type: "RESET" };

function parseFigure(fig: string | PlotParams): PlotParams {
  if (typeof fig !== "string") return fig;
  try {
    return JSON.parse(fig);
  } catch {
    return emptyFigure;
  }
}

const defaultSuccessData: SuccessResponseData = {
  fig: emptyFigure,
};

function successDataReducer(
  state: SuccessDataState,
  action: SuccessDataAction
): SuccessDataState {
  switch (action.type) {
    case "SET_SUCCESS":
      return action.payload;
    case "UPDATE_FIELD":
      return {
        ...state,
        [action.field]: action.value,
      };
    case "UPDATE_FIGURE":
      return {
        ...state,
        fig: parseFigure(action.value),
      };
    case "RESET":
      return defaultSuccessData;
    default:
      return state;
  }
}

export function useSuccessData() {
  const [successData, dispatch] = useReducer(
    successDataReducer,
    defaultSuccessData
  );

  const setSuccessData = useCallback((data: SuccessResponseData) => {
    dispatch({ type: "SET_SUCCESS", payload: data });
  }, []);

  const updateField = useCallback(
    <K extends keyof SuccessResponseData>(
      field: K,
      value: SuccessResponseData[K]
    ) => {
      dispatch({ type: "UPDATE_FIELD", field, value });
    },
    []
  );

  const updateFigure = useCallback((value: string | PlotParams) => {
    dispatch({ type: "UPDATE_FIGURE", value });
  }, []);

  const resetSuccessData = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  return {
    successData,
    setSuccessData,
    updateField,
    updateFigure,
    resetSuccessData,
  };
}
