import "@testing-library/jest-dom";
import { vi } from "vitest";


Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {
  configurable: true,
  value: vi.fn(),
});

Object.defineProperty(window, "scrollTo", {
  configurable: true,
  value: vi.fn(),
});


class ResizeObserverMock {
  observe() {}
  disconnect() {}
  unobserve() {}
}


Object.defineProperty(window, "ResizeObserver", {
  configurable: true,
  value: ResizeObserverMock,
});
