import { render, screen } from "@testing-library/react";
import SpotifyApp from "./SpotifyApp";

test("renders learn react link", () => {
  render(<SpotifyApp />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
