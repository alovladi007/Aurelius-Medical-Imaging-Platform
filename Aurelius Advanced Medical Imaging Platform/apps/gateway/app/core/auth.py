"""Authentication and authorization utilities."""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from keycloak import KeycloakOpenID
import os
from typing import List, Optional

# Keycloak configuration
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://keycloak:8080")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "aurelius")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "gateway")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET", "gateway-secret")

# Initialize Keycloak OpenID client
keycloak_openid = KeycloakOpenID(
    server_url=KEYCLOAK_URL,
    client_id=KEYCLOAK_CLIENT_ID,
    realm_name=KEYCLOAK_REALM,
    client_secret_key=KEYCLOAK_CLIENT_SECRET
)

security = HTTPBearer()


class User(BaseModel):
    """User model from JWT token."""
    sub: str
    username: str
    email: str
    roles: List[str]
    tenant_id: Optional[str] = None
    extra: dict = {}


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer token

    Returns:
        User: Authenticated user information

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        token = credentials.credentials

        # Verify and decode token
        userinfo = keycloak_openid.userinfo(token)

        # Extract roles
        token_info = keycloak_openid.introspect(token)
        roles = token_info.get("resource_access", {}).get(KEYCLOAK_CLIENT_ID, {}).get("roles", [])

        return User(
            sub=userinfo.get("sub"),
            username=userinfo.get("preferred_username"),
            email=userinfo.get("email", ""),
            roles=roles,
            tenant_id=userinfo.get("tenant_id"),
            extra=userinfo
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(required_role: str):
    """
    Dependency to require a specific role.

    Args:
        required_role: Role name required

    Returns:
        Dependency function
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if required_role not in user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return user
    return role_checker


def require_any_role(roles: List[str]):
    """
    Dependency to require any of the specified roles.

    Args:
        roles: List of acceptable roles

    Returns:
        Dependency function
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if not any(role in user.roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of roles {roles} required"
            )
        return user
    return role_checker
