"""Provides functions to easily load lyDATA CSV tables as :py:class:`pandas.DataFrame`.

The loading itself is implemented in the :py:class:`.LyDataset` class, which
is a :py:class:`pydantic.BaseModel` subclass. It validates the unique specification
that identifies a dataset and then allows loading it from the disk (if present) or
from GitHub (default).

The :py:func:`available_datasets` function can be used to create a generator of such
:py:class:`.LyDataset` instances, corresponding to all available datasets that
are either found on disk or on GitHub.

Consequently, the :py:func:`load_datasets` function can be used to load all datasets
matching the given specs/pattern. It takes the same arguments as the function
:py:func:`available_datasets` but returns a generator of :py:class:`pandas.DataFrame`
instead of :py:class:`.LyDataset`.

The docstring of all functions contains some basic doctest examples.
"""

import fnmatch
import warnings
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
from github import BadCredentialsException, Github, Repository, UnknownObjectException
from github.ContentFile import ContentFile
from github.GithubException import GithubException
from loguru import logger
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    PrivateAttr,
    RootModel,
    constr,
)

from lydata.accessor import LyDataFrame
from lydata.utils import get_github_auth
from lydata.validator import cast_dtypes, is_valid

_default_repo_name = "lycosystem/lydata"
low_min1_str = constr(to_lower=True, min_length=1)


class SkipDiskError(Exception):
    """Raised when the user wants to skip loading from disk."""


def _safely_fetch_repo(gh: Github, repo_name: str) -> Repository:
    """Fetch a GitHub repository, handling common errors."""
    try:
        logger.debug(f"Fetching repository '{repo_name}' from GitHub...")
        repo = gh.get_repo(repo_name)
    except UnknownObjectException as e:
        raise ValueError(f"Could not find repository '{repo_name}' on GitHub.") from e
    except BadCredentialsException as e:
        raise ValueError("Invalid GitHub credentials.") from e

    logger.debug(f"Fetched repository '{repo.full_name}' from GitHub.")
    return repo


def _safely_fetch_contents(
    repo: Repository,
    ref: str,
    path: str = ".",
) -> list[ContentFile] | ContentFile:
    """Fetch contents of a GitHub ``repo`` at a specific ``ref``, handling errors."""
    try:
        logger.debug(f"Fetching contents of repo '{repo.full_name}' at ref '{ref}'...")
        contents = repo.get_contents(path=path, ref=ref)
    except GithubException as e:
        available_branches = [b.name for b in repo.get_branches()]
        available_tags = [t.name for t in repo.get_tags()]
        raise ValueError(
            f"Could not find ref '{ref}' in repository '{repo.full_name}'.\n"
            f"Available branches: {available_branches}.\n"
            f"Available tags: {available_tags}."
        ) from e

    logger.debug(f"Fetched contents of repo '{repo.full_name}' at ref '{ref}'.")
    return contents


class LyDataset(BaseModel):
    """Specification of a dataset."""

    year: int = Field(
        gt=0,
        le=datetime.now().year,
        description="Release year of dataset.",
    )
    institution: low_min1_str = Field(
        description="Institution's short code. E.g., University Hospital Zurich: `usz`."
    )
    subsite: low_min1_str = Field(
        description="Tumor subsite(s) patients in this dataset were diagnosed with.",
    )
    repo_name: low_min1_str | None = Field(
        default=_default_repo_name,
        description="GitHub `repository/owner`.",
    )
    ref: low_min1_str | None = Field(
        default="main",
        description="Branch/tag/commit of the repo.",
    )
    local_dataset_dir: DirectoryPath | None = Field(
        default=None,
        description=(
            "Path to directory containing all the dataset subdirectories. So, e.g. if "
            "`path_on_disk` is `~/datasets` and the dataset is `2023-clb-multisite`, "
            "then the CSV file is expected to be at "
            "`~/datasets/2023-clb-multisite/data.csv`."
        ),
    )
    _content_file: ContentFile | None = PrivateAttr(default=None)

    @property
    def name(self) -> str:
        """Get the name of the dataset.

        >>> conf = LyDataset(year=2023, institution="clb", subsite="multisite")
        >>> conf.name
        '2023-clb-multisite'
        """
        return f"{self.year}-{self.institution}-{self.subsite}"

    def get_file_path(self) -> Path:
        """Get the path to the CSV dataset."""
        if self.local_dataset_dir is None:
            self.local_dataset_dir = Path(__file__).parent.parent

        dataset_path = self.local_dataset_dir / self.name / "data.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Could not find CSV locally at '{dataset_path}'.")

        logger.info(f"Found dataset {self.name} on disk at '{dataset_path}'.")
        return dataset_path

    def get_repo(
        self,
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> Repository:
        """Get the GitHub repository object.

        With the arguments ``token`` or ``user`` and ``password``, one can authenticate
        with GitHub. If no authentication is provided, the function will try to use the
        environment variables ``GITHUB_TOKEN`` or ``GITHUB_USER`` and
        ``GITHUB_PASSWORD``.

        >>> conf = LyDataset(
        ...     year=2021,
        ...     institution="clb",
        ...     subsite="oropharynx",
        ... )
        >>> conf.get_repo().full_name == conf.repo_name
        True
        >>> conf.get_repo().visibility
        'public'
        """
        auth = get_github_auth(token=token, user=user, password=password)
        gh = Github(auth=auth)
        return _safely_fetch_repo(gh=gh, repo_name=self.repo_name)

    def get_content_file(
        self,
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> ContentFile:
        """Get the GitHub content file of the data CSV.

        This method always tries to fetch the most recent version of the file.

        >>> conf = LyDataset(
        ...     year=2025,
        ...     institution="usz",
        ...     subsite="hypopharynx-larynx",
        ...     repo_name="lycosystem/lydata.private",
        ...     ref="2025-usz-hypopharynx-larynx",
        ... )
        >>> conf.get_content_file()
        ContentFile(path="2025-usz-hypopharynx-larynx/data.csv")
        """
        if self._content_file is not None:
            if self._content_file.update():
                logger.info(f"Content file of {self.name} was updated.")
            return self._content_file

        repo = self.get_repo(token=token, user=user, password=password)
        self._content_file = _safely_fetch_contents(
            repo=repo,
            path=f"{self.name}/data.csv",
            ref=self.ref,
        )
        return self._content_file

    def get_dataframe(
        self,
        use_github: bool = True,
        token: str | None = None,
        user: str | None = None,
        password: str | None = None,
        **load_kwargs,
    ) -> LyDataFrame:
        """Load the ``data.csv`` file from disk or from GitHub.

        One can also choose to ``use_github``. Any keyword arguments are passed to
        :py:func:`pandas.read_csv`.

        The method will store the output of :py:meth:`~pydantic.BaseModel.model_dump`
        in the :py:attr:`~pandas.DataFrame.attrs` attribute of the returned
        :py:class:`~pandas.DataFrame`.

        >>> conf = LyDataset(year=2021, institution="clb", subsite="oropharynx")
        >>> df = conf.get_dataframe(use_github=True)
        >>> df.shape
        (263, 82)
        """
        kwargs = {"header": [0, 1, 2]}
        kwargs.update(load_kwargs)

        if use_github:
            from_location = self.get_content_file(
                token=token, user=user, password=password
            ).download_url
        else:
            from_location = self.get_file_path()

        df = pd.read_csv(from_location, **kwargs)
        logger.info(f"Loaded dataset {self.name} from {from_location}.")
        df.attrs.update(self.model_dump())
        return df


def _available_datasets_on_disk(
    year: int | str = "*",
    institution: str = "*",
    subsite: str = "*",
    search_paths: list[Path] | None = None,
) -> Generator[LyDataset, None, None]:
    pattern = f"{str(year)}-{institution}-{subsite}"

    if search_paths is None:
        search_paths = [Path(__file__).parent.parent]

    search_paths = RootModel[list[DirectoryPath]].model_validate(search_paths).root

    for search_path in search_paths:
        for match in search_path.glob(pattern):
            if match.is_dir() and (match / "data.csv").exists():
                logger.debug(f"Found dataset directory at '{match}'.")
                year, institution, subsite = match.name.split("-", maxsplit=2)
                yield LyDataset(
                    year=year,
                    institution=institution,
                    subsite=subsite,
                    local_dataset_dir=search_path,
                    repo_name=None,
                    ref=None,
                )


def _available_datasets_on_github(
    year: int | str = "*",
    institution: str = "*",
    subsite: str = "*",
    repo_name: str = _default_repo_name,
    ref: str = "main",
) -> Generator[LyDataset, None, None]:
    """Generate :py:class:`.LyDataset` instances of available datasets on GitHub."""
    gh = Github(auth=get_github_auth())
    repo = _safely_fetch_repo(gh=gh, repo_name=repo_name)
    contents = _safely_fetch_contents(repo=repo, ref=ref)

    matches = []
    for content in contents:
        if content.type == "dir" and fnmatch.fnmatch(
            content.name, f"{year}-{institution}-{subsite}"
        ):
            matches.append(content)

    if len(matches) == 0:
        raise ValueError(
            f"No datasets found in repository '{repo_name}' matching "
            f"'{year}-{institution}-{subsite}' at ref '{ref}'."
        )

    for match in matches:
        year, institution, subsite = match.name.split("-", maxsplit=2)
        yield LyDataset(
            year=year,
            institution=institution,
            subsite=subsite,
            repo_name=repo.full_name,
            ref=ref,
        )


def available_datasets(
    year: int | str = "*",
    institution: str = "*",
    subsite: str = "*",
    search_paths: list[Path] | None = None,
    use_github: bool = True,
    repo_name: str = _default_repo_name,
    ref: str = "main",
) -> Generator[LyDataset, None, None]:
    """Generate :py:class:`.LyDataset` instances of available datasets.

    The arguments ``year``, ``institution``, and ``subsite`` represent glob patterns
    and all datasets matching these patterns can be iterated over using the returned
    generator.

    By default, the functions will look for datasets on the disk at paths specified
    in the ``search_paths`` argument. If no paths are provided, it will look in the
    the parent directory of the directory containing this file. If the library is
    installed, this will be the ``site-packages`` directory.

    With ``use_github`` set to ``True``, the function will not look for datasets on
    disk, but will instead look for them on GitHub. The ``repo`` and ``ref`` arguments
    can be used to specify the repository and the branch/tag/commit to look in.

    >>> avail_gen = available_datasets()
    >>> sorted([ds.name for ds in avail_gen])   # doctest: +NORMALIZE_WHITESPACE
    ['2021-clb-oropharynx',
     '2021-usz-oropharynx',
     '2023-clb-multisite',
     '2023-isb-multisite',
     '2025-hvh-oropharynx']
    >>> avail_gen = available_datasets(
    ...     repo_name="lycosystem/lydata.private",
    ...     ref="2025-umcg-hypopharynx-larynx",
    ...     use_github=True,
    ... )
    >>> sorted([ds.name for ds in avail_gen])   # doctest: +NORMALIZE_WHITESPACE
    ['2021-clb-oropharynx',
     '2021-usz-oropharynx',
     '2023-clb-multisite',
     '2023-isb-multisite',
     '2025-umcg-hypopharynx-larynx']
    >>> avail_gen = available_datasets(
    ...     institution="hvh",
    ...     ref="6ac98d",
    ...     use_github=True,
    ... )
    """
    if not use_github:
        if repo_name != _default_repo_name or ref != "main":
            warnings.warn(
                "Parameters `repo` and `ref` are ignored, unless `use_github` "
                "is set to `True`."
            )
        yield from _available_datasets_on_disk(
            year=year,
            institution=institution,
            subsite=subsite,
            search_paths=search_paths,
        )
    else:
        yield from _available_datasets_on_github(
            year=year,
            institution=institution,
            subsite=subsite,
            repo_name=repo_name,
            ref=ref,
        )


def load_datasets(
    year: int | str = "*",
    institution: str = "*",
    subsite: str = "*",
    search_paths: list[Path] | None = None,
    use_github: bool = True,
    repo_name: str = _default_repo_name,
    ref: str = "main",
    cast: bool = False,
    validate: bool = False,
    enhance: bool = False,
    **kwargs,
) -> Generator[LyDataFrame, None, None]:
    """Load matching datasets from GitHub or from the disk.

    It loads every dataset from the :py:class:`.LyDataset` instances generated by
    the :py:func:`available_datasets` function, which also receives most arguments of
    this function.

    The boolean flags ``cast``, ``validate``, and ``enhance`` can be used to
    automatically cast the dtypes of the loaded :py:class:`pandas.DataFrame`s,
    validate them, and enhance them with additional columns. These operations are
    performed using the :py:func:`~lydata.cast_dtypes`, :py:func:`~lydata.is_valid`,
    the :py:func:`~lydata.LyDataAccessor.enhance` method, respectively.

    Additional keyword arguments are passed to the :py:meth:`LyDataset.get_dataframe`
    method.
    """
    dset_confs = available_datasets(
        year=year,
        institution=institution,
        subsite=subsite,
        search_paths=search_paths,
        use_github=use_github,
        repo_name=repo_name,
        ref=ref,
    )
    for dset_conf in dset_confs:
        df: LyDataFrame = dset_conf.get_dataframe(use_github=use_github, **kwargs)
        df = cast_dtypes(df) if cast else df
        _ = validate and is_valid(df, fail_on_error=True)
        yield df.ly.enhance() if enhance else df
